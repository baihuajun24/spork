from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/serving/vllm024'))


class ContinuationTests(unittest.TestCase):
    def test_v4_continuation_is_explicit_and_does_not_mutate_messages(self):
        import vllm_v4_continue_patch as continuation
        class Tokenizer:
            def apply_chat_template(self, messages, tools=None, **kwargs):
                return kwargs.get('conversation', messages)
        native = types.ModuleType('vllm.tokenizers.deepseek_v4')
        native.get_deepseek_v4_tokenizer = lambda _: Tokenizer()
        package = types.ModuleType('vllm');package.__path__ = []
        tokenizers = types.ModuleType('vllm.tokenizers');tokenizers.__path__ = []
        package.tokenizers = tokenizers;tokenizers.deepseek_v4 = native
        with patch.dict(sys.modules, {'vllm': package, 'vllm.tokenizers': tokenizers,
                'vllm.tokenizers.deepseek_v4': native}), patch.object(continuation.importlib.metadata, 'version', return_value='0.24.0'):
            self.assertTrue(continuation.install_v4_continue_final_message_patch())
            self.assertFalse(continuation.install_v4_continue_final_message_patch())
            messages = [dict(role='assistant', content='prefix')]
            result = native.get_deepseek_v4_tokenizer(None).apply_chat_template(messages, continue_final_message=True)
            self.assertTrue(result[-1]['wo_eos'])
            self.assertNotIn('wo_eos', messages[-1])


if __name__ == '__main__':
    unittest.main()
