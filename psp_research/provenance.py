import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = 'dbbc4ee8ee66b6cd5286b939d7254fd7edcbfb8c'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def git_identity():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    return dict(upstream_commit=UPSTREAM, research_commit=git('rev-parse', 'HEAD'),
                working_tree_dirty=bool(git('status', '--porcelain')),
                loss_sha256=sha256(ROOT / 'loss.py'))


def check_upstream():
    subprocess.run(['python3', str(ROOT / 'scripts/check_upstream_integrity.py')], check=True)
