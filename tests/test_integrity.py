import ast
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import Runtime, dataset, evaluate, function_call, oracle_audit, tools_for
from experiment import extract, merge, parse_object, run_task, save
from study import build_manifest


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks, cls.gold = dataset()

    def test_all_reference_trajectories_pass(self):
        result = oracle_audit()
        self.assertEqual(result["invalid"], {})
        self.assertEqual(len(result["valid"]), 200)

    def test_reference_execution_matches_upstream(self):
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
        for key, task in self.tasks.items():
            env = Runtime(task)
            for turn in self.gold[key]:
                expected, _ = execute_multi_turn_func_call(turn, task["initial_config"], task["involved_classes"], "integrity_reference", key)
                actual = [env.execute(call) for call in turn]
                self.assertEqual(actual, expected, key)

    def test_empty_trajectory_fails(self):
        key = "multi_turn_base_0"
        self.assertFalse(evaluate(self.tasks[key], [[] for _ in self.gold[key]], self.gold[key])["valid"])

    def test_no_arbitrary_execution(self):
        env = Runtime(self.tasks["multi_turn_base_0"])
        for call in ["__import__('os').getcwd()", "open('README.md')", "eval('1+1')", "cat.__globals__"]:
            self.assertTrue(env.execute(call).startswith("Error during execution"))

    def test_literal_arguments_do_not_execute(self):
        expression = function_call("cat", {"file_name": "'); __import__('os').getcwd(); ('"}, {"cat"})
        tree = ast.parse(expression, mode="eval")
        self.assertIsInstance(tree.body.keywords[0].value, ast.Constant)

    def test_state_isolation(self):
        task = self.tasks["multi_turn_base_0"]
        before = copy.deepcopy(task)
        a, b = Runtime(task), Runtime(task)
        a.execute("mkdir(dir_name='new_directory')")
        self.assertNotEqual(a.execute("ls()"), b.execute("ls()"))
        self.assertEqual(task, before)

    def test_split_and_permutation_integrity(self):
        manifest = build_manifest()
        ids = [x for group in manifest["splits"].values() for x in group]
        self.assertEqual(len(ids), len(set(ids)))
        groups = [tuple(sorted(self.tasks[x]["path"])) for x in ids]
        self.assertEqual(len(groups), len(set(groups)))
        for order in manifest["orders"].values():
            self.assertEqual(set(order), set(manifest["splits"]["evolution"]))
            self.assertEqual(len(order), 30)

    def test_extractor_never_receives_gold_diagnostics(self):
        class Stub:
            def chat(self, messages):
                self.messages = messages
                return {"content": '{"items":[]}'}, {"finish_reason": "stop"}
        stub = Stub()
        task = self.tasks["multi_turn_base_0"]
        result = {"success": False, "messages": [{"role": "user", "content": "Observed request"}], "evaluation": {"secret_gold": "DO_NOT_LEAK"}}
        extract(task, result, stub)
        sent = json.dumps(stub.messages)
        self.assertNotIn("DO_NOT_LEAK", sent)
        self.assertNotIn("initial_config", sent)

    def test_gateway_thinking_wrapper(self):
        self.assertEqual(parse_object('<think>reasoning</think>\n```json\n{"items":[]}\n```'), {"items": []})

    def test_failed_checker_result_can_be_saved(self):
        import tempfile
        key = "multi_turn_base_0"
        wrong = copy.deepcopy(self.gold[key])
        wrong[0] = ["mkdir(dir_name='wrong')"]
        result = evaluate(self.tasks[key], [[turn] for turn in wrong], self.gold[key])
        self.assertFalse(result["valid"])
        with tempfile.TemporaryDirectory() as temp:
            save(Path(temp) / "failure.json", result)
            self.assertFalse(json.loads((Path(temp) / "failure.json").read_text(encoding="utf-8"))["valid"])

    def test_resume_does_not_repeat_completed_tool_calls(self):
        import tempfile
        task = copy.deepcopy(self.tasks["multi_turn_base_0"])
        task["question"] = task["question"][:1]
        reference = self.gold[task["id"]][:1]
        specs = [("cd", {"folder": "document"}), ("mkdir", {"dir_name": "temp"}), ("mv", {"source": "final_report.pdf", "destination": "temp"})]
        responses = [{"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "type": "function", "function": {"name": fn, "arguments": json.dumps(args)}}]} for i, (fn, args) in enumerate(specs)]
        responses.append({"role": "assistant", "content": "Done"})
        class Fake:
            model, temperature, max_tokens = "test", 0, 20
            def __init__(self, values):
                self.values = iter(values)
            def chat(self, messages, tools=None):
                value = next(self.values)
                if isinstance(value, Exception):
                    raise value
                return value, {"finish_reason": "tool_calls" if value.get("tool_calls") else "stop"}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.json"
            with self.assertRaises(RuntimeError):
                run_task(task, reference, Fake([responses[0], RuntimeError("transport")]), checkpoint_path=path)
            resumed = run_task(task, reference, Fake(responses[1:]), checkpoint_path=path)
            self.assertTrue(resumed["success"])
            self.assertEqual(sum(len(step) for turn in resumed["decoded"] for step in turn), 3)
            cached = run_task(task, reference, Fake([]), checkpoint_path=path)
            self.assertEqual(cached["decoded"], resumed["decoded"])

    def test_invalid_memory_is_not_resampled(self):
        class BadWriter:
            def chat(self, messages):
                return {"content": "not JSON"}, {"finish_reason": "stop"}
        items, log = extract(self.tasks["multi_turn_base_0"], {"messages": [], "success": False}, BadWriter())
        self.assertEqual(items, [])
        self.assertFalse(log["update_valid"])

    def test_local_embeddings_are_order_invariant(self):
        from embedding import embed
        import numpy as np
        texts = ["Check the current directory before moving a file", "Verify the account balance before placing an order"]
        a = embed(texts)
        b = embed(texts[::-1])
        self.assertTrue(np.array_equal(a[0], b[1]))
        self.assertEqual(len(a[0]), 768)
        prior = [{"content": "Keep a valid rule", "sources": ["t1"]}]
        result, log = merge(prior, [], BadWriter())
        self.assertEqual(result, prior)
        self.assertFalse(log["update_valid"])


if __name__ == "__main__":
    unittest.main()
