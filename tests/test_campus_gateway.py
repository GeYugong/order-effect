import unittest
import io
import json
import urllib.error
from unittest.mock import patch

from gateway import Gateway, BASE_URL, encode_payload
from study import build_manifest, client_for
from experiment import clean_message


class CampusGatewayTests(unittest.TestCase):
    def test_unicode_json_round_trip_preserves_all_payload_values(self):
        payload = {'messages': [{'role': 'user', 'content': 'then select the item from the list; 中文 😀 × " \\ \n'}], 'bool': True, 'none': None, 'temperature': 1.0, 'tools': [{'name': 'find', 'arguments': '{"x": "\\u1234"}'}]}
        encoded = encode_payload(payload)
        self.assertEqual(json.loads(encoded), payload)
        self.assertNotIn(b'select', encoded)

    def test_524_retries_identical_payload_with_cooldown(self):
        payload = {'model': 'DeepSeek-V4-Flash-0731-W8A8', 'messages': [{'role': 'user', 'content': 'hi'}]}
        error = urllib.error.HTTPError(BASE_URL, 524, 'timeout', {'Retry-After': '90'}, io.BytesIO(b'upstream timeout'))
        with patch('gateway.api_key', return_value='test-key'), patch('gateway.rate_limit'), patch('gateway.time.sleep') as sleep, patch('gateway.urllib.request.build_opener') as factory:
            opener = factory.return_value
            opener.open.side_effect = [error, io.BytesIO(b'{"model":"campus","choices":[]}')]
            client = Gateway(payload['model'])
            client.request('POST', '/chat/completions', payload)
            requests = [c.args[0] for c in opener.open.call_args_list]
            self.assertEqual(requests[0].data, requests[1].data)
            self.assertEqual(json.loads(requests[0].data), payload)
            sleep.assert_called_once_with(90)
            self.assertEqual(factory.call_args.args[0].proxies, {})

    def test_auth_errors_are_not_retried(self):
        error = urllib.error.HTTPError(BASE_URL, 401, 'unauthorized', {}, io.BytesIO(b'bad key'))
        with patch('gateway.api_key', return_value='test-key'), patch('gateway.rate_limit'), patch('gateway.time.sleep') as sleep, patch('gateway.urllib.request.build_opener') as factory:
            factory.return_value.open.side_effect = error
            with self.assertRaisesRegex(RuntimeError, '401'):
                Gateway('campus').request('POST', '/chat/completions', {})
            self.assertEqual(factory.return_value.open.call_count, 1)
            sleep.assert_not_called()

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
