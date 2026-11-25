#!/usr/bin/env python3
"""Verify original research source files without importing ML dependencies."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
record = json.loads((root / 'docs/UPSTREAM_SNAPSHOT.json').read_text())
locked = ['LICENSE', 'README.md', 'main.py', 'PSP_UAP.py', 'loss.py',
          'functions.py', 'semantic_aug.py', 'strategy.py']
failed = []
for name in locked:
    p = root / name
    expected = record['files'][name]['sha256']
    actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    ok = actual == expected
    print(('PASS' if ok else 'FAIL') + ' ' + name)
    if not ok:
        failed.append(name)
if failed:
    raise SystemExit('Upstream source drift: ' + ', '.join(failed))
print('Original loss and upstream source intact; this does not validate training results.')
