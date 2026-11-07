"""Explicit model setup shared by command-line examples.

The auto family retains the original Qwen adapter selection. Qwen3.5 JSON
rendering is an explicit choice because its tokenizer/template must agree.
"""
from .deepseek_v4 import DeepSeekV4ToolCallAdapter
from .glm53 import Glm53ToolCallAdapter
from .qwen import QwenToolCallAdapter, Qwen35ToolCallAdapter

MODEL_FAMILIES = ('auto', 'qwen', 'qwen35', 'deepseek_v4', 'glm53')


def make_adapter(family, tokenizer, tools, model_name=''):
    if family == 'deepseek_v4':
        return DeepSeekV4ToolCallAdapter(tokenizer, tools)
    if family == 'glm53':
        return Glm53ToolCallAdapter(tokenizer, tools)
    if family not in MODEL_FAMILIES:
        raise ValueError(f'unknown model family: {family}')
    legacy_xml = family == 'auto' and ('3.5' in model_name or 'qwen3.5' in model_name.lower())
    cls = Qwen35ToolCallAdapter if legacy_xml else QwenToolCallAdapter
    return cls(tokenizer, tools)


def configure_client(client, family, *, enable_thinking=True):
    name = client.model.lower()
    if family in ('deepseek_v4', 'glm53'):
        client.main_stop_token_ids = ()
        client.include_reasoning = True
        client.parallel_tool_calls = False
        client.chat_template_kwargs = {'thinking': bool(enable_thinking)}
        if family == 'glm53':
            client.chat_template_kwargs['clear_thinking'] = True
            if enable_thinking:
                client.chat_template_kwargs['reasoning_effort'] = 'max'
    elif family == 'qwen35' or (family == 'auto' and
            ('3.5' in name or ('a3b' in name and '35' in name))):
        client.main_stop_token_ids = (248058,)
    elif family == 'auto' and 'phi' in name:
        client.main_stop_token_ids = ()
    return client


def probe_targets(value):
    """Parse strictly increasing positive output-stream trigger depths."""
    targets = tuple(int(part.strip()) for part in value.split(','))
    if not targets or targets[0] < 1 or any(a >= b for a, b in zip(targets, targets[1:])):
        raise ValueError('probe targets must be increasing positive integers, e.g. 1,16')
    return targets
