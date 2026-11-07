import json
import unittest
from spork_core.vllm_client import VllmClient


class Response:
    status = 200

    def __init__(self, chat):
        choices = [{'delta': {'reasoning': 'inspect'}}] if chat else [{'text': 'x'}]
        chunks = [{'choices': choices}]
        if chat:
            chunks.append({'choices': [{'delta': {'content': 'done'}}]})
        chunks.append({'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 2,
            'total_tokens': 12, 'prompt_tokens_details': {'cached_tokens': 7}}})
        self.lines = [b'data: ' + json.dumps(item).encode() for item in chunks] + [b'data: [DONE]']
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    def raise_for_status(self):
        pass

    async def json(self):
        return dict(choices=[{'message': {'content': 'done'}}],
                    usage={'prompt_tokens': 10, 'completion_tokens': 2})

    def __aiter__(self):
        self.iterator = iter(self.lines)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopAsyncIteration


class Session:
    def post(self, url, *, json, timeout):
        self.body = json
        return Response('/chat/' in url)


class UsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_completion_usage_survives_empty_choices(self):
        session = Session()
        row = await VllmClient(model='fixture').stream(session, 'prompt', max_tokens=2)
        self.assertEqual(session.body['stream_options'], {'include_usage': True})
        self.assertEqual(row['text'], 'x')
        self.assertEqual(row['usage']['prompt_tokens_details']['cached_tokens'], 7)

    async def test_reasoning_triggers_probe_callback_and_usage_is_retained(self):
        session, deltas = Session(), []
        async def callback(text, count):
            deltas.append((text, count))
        row = await VllmClient(model='fixture', include_reasoning=True).stream_chat(
            session, [{'role': 'user', 'content': 'fixture'}], max_tokens=2, on_token=callback)
        self.assertEqual(deltas, [('inspect', 1), ('done', 2)])
        self.assertEqual(row['reasoning'], 'inspect')
        self.assertEqual(row['text'], 'done')
        self.assertEqual(row['usage']['completion_tokens'], 2)

    async def test_nonstreaming_chat_probe_retains_usage(self):
        row, _ = await VllmClient(model='fixture').complete_chat(
            Session(), [{'role': 'user', 'content': 'fixture'}], max_tokens=2)
        self.assertEqual(row['usage']['completion_tokens'], 2)


if __name__ == '__main__':
    unittest.main()
