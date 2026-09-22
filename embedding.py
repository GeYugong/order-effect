"""Fixed local CPU embeddings; no network service in the experimental loop."""
import hashlib
import json
from pathlib import Path
import threading

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "BAAI/bge-base-en-v1.5"
_lock = threading.RLock()
_model = None
_cache = {}


def embed(texts, query=False):
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding
            _model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(ROOT / ".cache/embedding"), threads=2, local_files_only=(ROOT / "embedding.lock.json").exists())
        missing = list(dict.fromkeys(text for text in texts if (query, text) not in _cache))
        if missing:
            vectors = list(_model.query_embed(missing) if query else _model.passage_embed(missing))
            for text, vector in zip(missing, vectors):
                _cache[(query, text)] = vector.tolist()
        return [_cache[(query, text)][:] for text in texts]


def model_lock():
    sample = embed(["tool-use memory"])
    files = {}
    for path in sorted((ROOT / ".cache/embedding").rglob("*")):
        if path.is_file() and path.suffix in (".onnx", ".json", ".txt"):
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(block)
            files[str(path.relative_to(ROOT / ".cache/embedding"))] = h.hexdigest()
    result = {"model": MODEL_NAME, "dimensions": len(sample[0]), "threads": 2, "backend": "fastembed CPU ONNX", "sha256": files}
    path = ROOT / "embedding.lock.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != result:
        raise RuntimeError("Local embedding model does not match embedding.lock.json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    locked = model_lock()
    print(json.dumps({"model": locked["model"], "dimensions": locked["dimensions"], "files": len(locked["sha256"])}))
