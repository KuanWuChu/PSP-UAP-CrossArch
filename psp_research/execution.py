"""Evidence gates for full research runs; no target-result checkpoint selection."""
import json
from pathlib import Path
from .provenance import UPSTREAM, sha256
from .registry import E0_NAMES


def require_baseline(path, manifest_hashes):
    if not path:
        raise ValueError('full training requires a verified --baseline-record')
    r = json.loads(Path(path).read_text())
    if (r.get('status') != 'complete' or r.get('upstream_commit') != UPSTREAM or
        r.get('baseline_checks_passed') is not True or r.get('manifest_hashes') != manifest_hashes or
        r.get('execution_label') not in {'upstream_modern_compatibility','upstream_legacy_core'}):
        raise ValueError('baseline evidence/source/data validation not complete')
    if set(r.get('targets',{})) != set(E0_NAMES):
        raise ValueError('baseline must include the original five CNN targets')
    for metrics in r['targets'].values():
        if (metrics.get('total_count') != 50000 or
            metrics.get('fr') != metrics.get('flip_count',-1)/50000 or
            metrics.get('clean_acc') != metrics.get('clean_correct_count',-1)/50000):
            raise ValueError('baseline raw counts/metrics mismatch')
    if sha256(r['uap_path']) != r['uap_sha256']:
        raise ValueError('baseline UAP content changed')
    return dict(baseline_record_sha256=sha256(path),baseline_uap_sha256=r['uap_sha256'],
                baseline_execution_label=r['execution_label'])
