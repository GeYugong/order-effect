"""Download a pinned, minimal subset of the official BFCL code and data."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO = "ShishirPatil/gorilla"
PREFIX = "berkeley-function-call-leaderboard/"


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "order-effect-research"}), timeout=60) as r:
        return r.read()


def main():
    lock_path = ROOT / "source.lock.json"
    old = json.loads(lock_path.read_text()) if lock_path.exists() else None
    ref = old["commit"] if old else "main"
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{ref}?recursive=1"))
    commit = old["commit"] if old else json.loads(fetch(f"https://api.github.com/repos/{REPO}/commits/main"))["sha"]
    # Fetch tree again at the resolved commit so metadata and downloads cannot race main.
    if not old:
        tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1"))
    exact = {
        "LICENSE", PREFIX + "README.md", PREFIX + "bfcl_eval/__init__.py",
        PREFIX + "bfcl_eval/constants/__init__.py",
        PREFIX + "bfcl_eval/constants/executable_backend_config.py",
        PREFIX + "bfcl_eval/eval_checker/__init__.py",
        PREFIX + "bfcl_eval/data/BFCL_v4_multi_turn_base.json",
        PREFIX + "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json",
        PREFIX + "bfcl_eval/data/README.md",
    }
    starts = (PREFIX + "bfcl_eval/eval_checker/multi_turn_eval/", PREFIX + "bfcl_eval/data/multi_turn_func_doc/")
    paths = [e["path"] for e in tree["tree"] if e["type"] == "blob" and (e["path"] in exact or e["path"].startswith(starts))]
    def download(path):
        relative = path.removeprefix(PREFIX)
        dest = ROOT / "vendor" / relative
        expected = old["sha256"].get(relative) if old else None
        data = dest.read_bytes() if dest.exists() else fetch(f"https://raw.githubusercontent.com/{REPO}/{commit}/{path}")
        checksum = hashlib.sha256(data).hexdigest()
        if expected and checksum != expected:
            raise RuntimeError(f"Source checksum mismatch: {relative}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
        return relative, checksum
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        checksums = dict(pool.map(download, paths))
    lock = {"repository": f"https://github.com/{REPO}", "commit": commit, "sha256": dict(sorted(checksums.items()))}
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"commit": commit, "files": len(checksums)}))


if __name__ == "__main__":
    main()
