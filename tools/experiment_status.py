"""Read local experiment progress without changing or restarting any experiment."""
import _bootstrap
import argparse
import json
import os
import re
from pathlib import Path


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return None


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='outputs/full-experiments-20261126')
    a=p.parse_args();root=Path(a.root)
    result=dict(github_publication_paused=(read(root/'publication-paused.json') or {}).get('publication_paused',False),job=read(root/'job-status.json'),baseline_verified=(read(root/'baseline-validated.json') or {}).get('baseline_checks_passed',False),publication=read(root/'publication.json'))
    try:
        pid=int((root/'job.pid').read_text());os.kill(pid,0)
        result['controller_process_exists']=True
    except (OSError,ValueError):result['controller_process_exists']=False
    native=root/'e0-legacy-original/native.log'
    if native.exists():
        with native.open('rb') as f:f.seek(max(0,native.stat().st_size-4096));text=f.read().decode(errors='replace')
        progress=re.findall(r'(\d+)/10000',text)
        result['legacy_last_logged_progress']=int(progress[-1]) if progress else None
    matrix=read(root/'matrix/status.json')
    if matrix:
        result['matrix']=dict(status=matrix['status'],phase=matrix['phase'],current_run=matrix.get('current_run'),
            complete_training_runs=sum(x['status']=='complete' for x in matrix['runs'].values()),
            failed_training_runs=sum(x['status']=='failed' for x in matrix['runs'].values()),aggregate=matrix.get('aggregate'))
    print(json.dumps(result,indent=2))
