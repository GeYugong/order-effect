import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mechanism import factorial_job, repair_candidate
from mechanism_report import risk_difference


class FakeClient:
    def __init__(self, items):
        self.items = items
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return {'content': json.dumps({'items': self.items})}, {'finish_reason': 'stop'}


class MechanismTests(unittest.TestCase):
    def setUp(self):
        self.before = [{'content':'old advice','sources':['old']}]
        self.incoming = [{'source':'new','content':'new experience'}]
        self.candidate = [{'content':'x '*101, 'sources':['new']}, {'content':'unchanged','sources':['old']}]
        self.generation = {'error':'Playbook rule exceeds word budget','message':{'content':json.dumps({'rules':self.candidate})}}

    def test_only_long_content_changes_sources_and_other_rules_unchanged(self):
        client = FakeClient([{'index':0,'content':'short rule'}])
        rules, log = repair_candidate(self.before,self.incoming,self.generation,client)
        self.assertTrue(log['accepted'])
        self.assertEqual(rules[0]['sources'], ['new'])
        self.assertEqual(rules[1], self.candidate[1])
        self.assertEqual(self.before[0]['content'], 'old advice')
        self.assertEqual(len(client.calls), 1)
        payload = json.loads(client.calls[0][1]['content'])
        self.assertEqual(set(payload), {'items'})
        self.assertNotIn('success', payload)

    def test_second_invalid_repair_falls_back_without_resampling(self):
        client=FakeClient([{'index':0,'content':'still '*101}])
        rules,log=repair_candidate(self.before,self.incoming,self.generation,client)
        self.assertFalse(log['accepted']);self.assertEqual(rules,self.before);self.assertEqual(len(client.calls),1)

    def test_provenance_invalid_candidate_is_not_repaired(self):
        candidate=copy.deepcopy(self.candidate);candidate[1]['sources']=['invented']
        generation={**self.generation,'message':{'content':json.dumps({'rules':candidate})}}
        client=FakeClient([]);rules,log=repair_candidate(self.before,self.incoming,generation,client)
        self.assertFalse(log['attempted']);self.assertEqual(client.calls,[]);self.assertEqual(rules,self.before)

    def test_wrong_repair_index_is_rejected(self):
        client=FakeClient([{'index':1,'content':'wrong target'}])
        rules,log=repair_candidate(self.before,self.incoming,self.generation,client)
        self.assertFalse(log['accepted']);self.assertEqual(rules,self.before)

    def test_malformed_rule_is_skipped_instead_of_crashing_stage(self):
        generation={**self.generation,'message':{'content':json.dumps({'rules':[None]})}}
        client=FakeClient([]);rules,log=repair_candidate(self.before,self.incoming,generation,client)
        self.assertEqual(rules,self.before);self.assertFalse(log['attempted']);self.assertEqual(client.calls,[])

    def test_completed_proposal_and_task_are_reused_on_resume(self):
        x={'model_config':{'orders':{'shuffle1':['source']},'splits':{'evaluation':['task']},'max_steps_per_turn':12}}
        job={'policy':'skip_baseline','order':'shuffle1','rep':0}
        pool={'experiences':{'source':[{'content':'experience','source':'source'}]}}
        client=SimpleNamespace(usage={'total_tokens':0},set_ledger=lambda _:None)
        with tempfile.TemporaryDirectory() as tmp, patch('mechanism.DATA',Path(tmp)), patch('mechanism.client_for',return_value=client), patch('mechanism.dataset',return_value=({'task':{}},{'task':[]})), patch('mechanism.merge',return_value=([{'content':'a rule','sources':['source']}],{'update_valid':True})) as merge, patch('mechanism.run_task',return_value={'success':True}) as task:
            result=factorial_job(job,x,pool)
            self.assertEqual(result['score'],1)
            folder=Path(tmp)/'factorial/skip_baseline_shuffle1_r0'
            # Simulate a recovered process with a saved proposal and task but no summary.
            (folder/'summary.json').unlink()
            (folder/'merge_000.json').unlink()
            resumed=factorial_job(job,x,pool)
            self.assertEqual(resumed['memory_hash'],result['memory_hash'])
            self.assertEqual(merge.call_count,1);self.assertEqual(task.call_count,1)

    def test_exact_difference_interval_handles_boundary_counts(self):
        lo,hi=risk_difference([1]*10,[0]*10)
        self.assertGreater(lo,0);self.assertEqual(hi,1)
        lo,hi=risk_difference([0]*10,[0]*10)
        self.assertLess(lo,0);self.assertGreater(hi,0)
