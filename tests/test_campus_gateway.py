import unittest
from unittest.mock import patch

from gateway import Gateway, BASE_URL
from study import build_manifest, client_for
from experiment import clean_message


class CampusGatewayTests(unittest.TestCase):
    def test_paid_route_is_blocked_before_credentials_or_network(self):
        with patch('gateway.api_key') as key, patch('gateway.urllib.request.urlopen') as network:
            for model in ('deepseek-flash', 'deepseek-v4-pro'):
                with self.assertRaisesRegex(RuntimeError, 'disabled'):
                    Gateway(model).request('POST', '/chat/completions', {})
            key.assert_not_called()
            network.assert_not_called()

    def test_campus_payload_and_reasoning_round_trip(self):
        client = client_for(build_manifest())
        response = {'model': client.model, 'choices': [{'message': {'role': 'assistant', 'content': 'done', 'reasoning': 'observed reasoning'}, 'finish_reason': 'stop'}]}
        with patch.object(client, 'request', return_value=response) as request:
            message, meta = client.chat([{'role': 'user', 'content': 'test'}])
        payload = request.call_args.args[2]
        self.assertEqual(payload['max_tokens'], 32768)
        self.assertEqual(payload['chat_template_kwargs']['thinking_mode'], 'thinking')
        self.assertEqual(payload['chat_template_kwargs']['reasoning_effort'], 'high')
        self.assertEqual(clean_message(message)['reasoning_content'], 'observed reasoning')
        self.assertEqual(meta['endpoint'], BASE_URL)
