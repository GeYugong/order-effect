"""BFCL adapters: official simulators/checker, with literal-only tool execution."""
import ast
import copy
import importlib
import json
from pathlib import Path
import sys
import threading
import types

ROOT = Path(__file__).resolve().parent
_runtime_lock = threading.RLock()
sys.path.insert(0, str(ROOT / "vendor"))
from bfcl_eval.constants.executable_backend_config import (  # noqa: E402
    CLASS_FILE_PATH_MAPPING, MULTI_TURN_FUNC_DOC_FILE_MAPPING, STATELESS_CLASSES,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def dataset():
    base = ROOT / "vendor/bfcl_eval/data"
    tasks = {t["id"]: t for t in read_jsonl(base / "BFCL_v4_multi_turn_base.json")}
    gold = {t["id"]: t["ground_truth"] for t in read_jsonl(base / "possible_answer/BFCL_v4_multi_turn_base.json")}
    return tasks, gold


def schema(node):
    if isinstance(node, list):
        return [schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    result = {k: schema(v) for k, v in node.items() if k != "default"}
    mapping = {"dict": "object", "float": "number", "int": "integer", "bool": "boolean", "tuple": "array", "any": "string"}
    if isinstance(result.get("type"), str):
        result["type"] = mapping.get(result["type"], result["type"])
    return result


def tools_for(task):
    functions = []
    for cls in task["involved_classes"]:
        doc = ROOT / "vendor/bfcl_eval/data/multi_turn_func_doc" / MULTI_TURN_FUNC_DOC_FILE_MAPPING[cls]
        for f in read_jsonl(doc):
            if f["name"] not in task.get("excluded_function", []):
                functions.append({"type": "function", "function": {
                    "name": f["name"], "description": f["description"], "parameters": schema(f["parameters"]),
                }})
    assert len({f["function"]["name"] for f in functions}) == len(functions)
    return functions


class Runtime:
    def __init__(self, task):
        self.instances = {}
        self.methods = {}
        for name in task["involved_classes"]:
            module = importlib.import_module(CLASS_FILE_PATH_MAPPING[name])
            instance = getattr(module, name)()
            if name not in STATELESS_CLASSES:
                instance._load_scenario(copy.deepcopy(task["initial_config"].get(name, {})), long_context=False)
            self.instances[name] = instance
            # Only documented methods can be called; no Python, shell, network or host filesystem tools.
            doc = ROOT / "vendor/bfcl_eval/data/multi_turn_func_doc" / MULTI_TURN_FUNC_DOC_FILE_MAPPING[name]
            for fn in read_jsonl(doc):
                self.methods[fn["name"]] = getattr(instance, fn["name"])

    def _value(self, node):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in self.methods:
                raise ValueError("Only documented simulator functions are allowed")
            if any(k.arg is None for k in node.keywords):
                raise ValueError("Argument unpacking is not allowed")
            return self.methods[node.func.id](
                *[self._value(v) for v in node.args],
                **{k.arg: self._value(k.value) for k in node.keywords},
            )
        # literal_eval rejects imports, attributes, comprehensions, and arbitrary execution.
        return ast.literal_eval(node)

    def execute(self, call):
        # Upstream MathAPI changes process-global mpmath precision. Serialize the very
        # short simulator calls (including result formatting), not network/model calls.
        with _runtime_lock:
            return self._execute_locked(call)

    def _execute_locked(self, call):
        try:
            node = ast.parse(call, mode="eval").body
            if not isinstance(node, ast.Call):
                raise ValueError("Expected one tool call")
            result = self._value(node)
            if isinstance(result, str):
                return result
            if isinstance(result, dict):
                try:
                    return json.dumps(result)
                except (TypeError, ValueError):
                    return str(result)
            return str(result)
        except Exception as exc:
            return f"Error during execution: {exc}"


def function_call(name, arguments, allowed):
    if name not in allowed:
        raise ValueError("Undocumented or excluded tool")
    if not isinstance(arguments, dict) or any(not k.isidentifier() for k in arguments):
        raise ValueError("Tool arguments must be an object with identifier keys")
    return name + "(" + ", ".join(k + "=" + repr(v) for k, v in arguments.items()) + ")"


def evaluate(task, decoded, gold):
    runtimes = {}
    def executor(func_call_list, initial_config, involved_classes, model_name, test_entry_id, **kwargs):
        # Separate model/gold instances; no state survives a task or run.
        if model_name not in runtimes:
            runtimes[model_name] = Runtime(task)
        env = runtimes[model_name]
        return [env.execute(call) for call in func_call_list], env.instances
    # Retain official comparison logic, replacing only its unsafe eval-based executor.
    checker = types.FunctionType(multi_turn_checker.__code__, {
        **multi_turn_checker.__globals__, "execute_multi_turn_func_call": executor,
    })
    return checker(decoded, gold, task, "multi_turn_base", "candidate")


def oracle_audit():
    tasks, answers = dataset()
    report = {"total": len(tasks), "valid": [], "invalid": {}}
    for task_id, task in tasks.items():
        try:
            gold = answers[task_id]
            # Replay each reference turn in a separate model and reference environment.
            result = evaluate(task, [[turn] for turn in gold], gold)
            if result["valid"]:
                report["valid"].append(task_id)
            else:
                report["invalid"][task_id] = result.get("error_type", "invalid")
        except Exception as exc:
            report["invalid"][task_id] = type(exc).__name__ + ": " + str(exc)
    return report


if __name__ == "__main__":
    report = oracle_audit()
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/oracle_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"total": report["total"], "valid": len(report["valid"]), "invalid": report["invalid"]}))
