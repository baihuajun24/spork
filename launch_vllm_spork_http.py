#!/usr/bin/env python3
"""SPORK-enabled vLLM HTTP server launcher.

Patches GPUWorker with spork_set_draft_tokens() before the engine core process
forks, then adds a /spork/set_tokens sidecar endpoint to the API server.

Usage:
    CUDA_VISIBLE_DEVICES=2 /usr/bin/python3 coder/scripts/launch_vllm_spork_http.py \
        --model <path> --port 8100 --tensor-parallel-size 1 \
        --speculative-model [ngram] --num-speculative-tokens 20 \
        --ngram-prompt-lookup-max 20 --max-model-len 32768 \
        --gpu-memory-utilization 0.85

The sidecar endpoint /spork/set_tokens accepts:
    POST {"request_id": "...", "draft_tokens": [...], "prompt_len": 0}
and routes them to the SporkProposer inside the engine core process via collective_rpc.
"""
import os
import sys
from pathlib import Path
import json

# MUST be before any CUDA import to keep fork mode
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

# ── Step 1: Monkey-patch GPUWorker BEFORE any vLLM process forks ──

sys.path.insert(0, str(Path(__file__).parent))
from spork_core.spork_proposer import SporkProposer, patch_drafter

from vllm.v1.worker.gpu_worker import Worker as GPUWorker


def _spork_set_draft_tokens(self, request_id: str, draft_tokens: list[int],
                            prompt_len: int = 0) -> dict:
    """Called via collective_rpc from the HTTP server process."""
    runner = self.model_runner
    drafter = getattr(runner, "drafter", None)
    if drafter is None:
        return {
            "status": "error",
            "error": "no_drafter",
            "message": "vLLM was launched without speculative decoding; D3 draft injection is unavailable.",
        }

    # Lazy-init: first call patches the drafter with SporkProposer
    if not hasattr(drafter, '_spork'):
        boundary_token_ids = []
        raw_boundary = os.environ.get("SPORK_BOUNDARY_TOKEN_IDS", "").strip()
        if raw_boundary:
            boundary_token_ids = json.loads(raw_boundary)
        proposer = SporkProposer(
            original_drafter=drafter,
            boundary_token_ids=boundary_token_ids or None,
        )
        patch_drafter(runner, proposer)
        print(f"[SPORK HTTP] SporkProposer patched on worker (lazy init)", flush=True)

    proposer = drafter._spork
    proposer.clear_all()
    proposer.enabled = True
    proposer.set_probe_tokens(0, draft_tokens, prompt_len=prompt_len)
    return {"status": "ok", "n_tokens": len(draft_tokens), "request_id": request_id}


def _spork_get_status(self) -> dict:
    """Return SporkProposer stats via collective_rpc."""
    runner = self.model_runner
    drafter = getattr(runner, "drafter", None)
    if drafter is None:
        return {"patched": False, "has_drafter": False}
    if not hasattr(drafter, '_spork'):
        return {"patched": False, "has_drafter": True}
    proposer = drafter._spork
    snap = proposer.snapshot()
    snap["patched"] = True
    snap["enabled"] = proposer.enabled
    snap["fired"] = proposer._fired
    snap["has_probe_tokens"] = proposer._probe_tokens is not None
    if proposer.last_inject_info:
        snap["last_inject_info"] = dict(proposer.last_inject_info)
    return snap


def _spork_clear(self) -> dict:
    """Clear SporkProposer state via collective_rpc."""
    runner = self.model_runner
    drafter = getattr(runner, "drafter", None)
    if drafter is None:
        return {"status": "cleared", "has_drafter": False}
    if hasattr(drafter, '_spork'):
        drafter._spork.clear_all()
        drafter._spork.enabled = False
    return {"status": "cleared", "has_drafter": True}


# Patch the class — inherited by forked engine core process
GPUWorker.spork_set_draft_tokens = _spork_set_draft_tokens
GPUWorker.spork_get_status = _spork_get_status
GPUWorker.spork_clear = _spork_clear
print("[SPORK HTTP] GPUWorker patched with spork methods", flush=True)


# ── Step 2: Add sidecar routes to the API server ──
#
# PORT NOTE (2026-06-14): vLLM 0.19.1/0.20.2 removed the module-level
# `api_server.router` object that the legacy code attached routes to. Routes are
# now registered onto the FastAPI `app` that `build_app(args, ...)` constructs
# and returns. We monkey-patch `build_app` to wrap the original, then add the
# /spork/* routes directly onto the returned app via `app.add_api_route`.
# `app.state.engine_client` is populated later by `init_app_state`, so the
# handlers read it lazily from `raw_request.app.state` at request time.

from vllm.entrypoints.openai import api_server as _api_server
from fastapi import Request
from fastapi.responses import JSONResponse
import msgspec


class SetTokensRequest(msgspec.Struct):
    request_id: str = ""
    draft_tokens: list[int] = []
    prompt_len: int = 0


async def spork_set_tokens(raw_request: Request):
    body = await raw_request.body()
    req = msgspec.json.decode(body, type=SetTokensRequest)
    engine_client = raw_request.app.state.engine_client
    results = await engine_client.collective_rpc(
        "spork_set_draft_tokens",
        args=(req.request_id, req.draft_tokens, req.prompt_len),
    )
    return JSONResponse(content=results[0] if results else {"error": "no workers"})


async def spork_status(raw_request: Request):
    engine_client = raw_request.app.state.engine_client
    results = await engine_client.collective_rpc("spork_get_status")
    return JSONResponse(content=results[0] if results else {"error": "no workers"})


async def spork_clear(raw_request: Request):
    engine_client = raw_request.app.state.engine_client
    results = await engine_client.collective_rpc("spork_clear")
    return JSONResponse(content=results[0] if results else {"error": "no workers"})


_orig_build_app = _api_server.build_app


def _build_app_with_spork(*args, **kwargs):
    app = _orig_build_app(*args, **kwargs)
    app.add_api_route("/spork/set_tokens", spork_set_tokens, methods=["POST"])
    app.add_api_route("/spork/status", spork_status, methods=["GET"])
    app.add_api_route("/spork/clear", spork_clear, methods=["POST"])
    print("[SPORK HTTP] Sidecar routes registered on app: "
          "/spork/set_tokens, /spork/status, /spork/clear", flush=True)
    return app


_api_server.build_app = _build_app_with_spork
print("[SPORK HTTP] build_app patched to add sidecar routes", flush=True)


# ── Step 3: Launch the vLLM server ──

if __name__ == "__main__":
    import uvloop
    from vllm.entrypoints.openai.api_server import (
        run_server, make_arg_parser, validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.utils import cli_env_setup

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="SPORK-enabled vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))
