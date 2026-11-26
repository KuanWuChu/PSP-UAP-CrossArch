"""Run the prespecified15-run matrix and90 target evaluations after verified E0.

Restart reuses only complete, identity-matching artifacts and resumes interrupted
training from its latest immutable checkpoint. Failed numerical runs remain failed.
"""
import _bootstrap
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from psp_research.aggregate import aggregate, validate_metric, PILOT_TARGETS
from psp_research.config import load_config
from psp_research.data import verify_bundle
from psp_research.execution import require_baseline
from psp_research.provenance import ROOT, git_identity, sha256, digest, write_json, check_upstream


def execute(a):
    matrix=json.loads(Path(a.matrix).read_text())
    configs={r['variant']:load_config(r['config']) for r in matrix['runs']}
    if len(matrix['runs'])!=15 or any(c['max_steps']!=10000 for c in configs.values()):
        raise ValueError('expected prespecified15 runs /10000 steps')
    identity=dict(configs=configs,matrix_sha256=sha256(a.matrix),
        weights_lock_sha256=sha256(a.weights_lock),targets=PILOT_TARGETS,
        determinism='warn',precision='float32')
    if a.dry_run:
        print(json.dumps(identity,indent=2));return
    check_upstream()
    index=verify_bundle(a.manifest_dir)
    identity.update(manifest_hashes=index['manifest_hashes'],
                    **require_baseline(a.baseline_record,index['manifest_hashes']))
    git=git_identity()
    if git['working_tree_dirty']:raise ValueError('commit source before starting full matrix')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'evaluations').mkdir(exist_ok=True)
    with (out/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity['research_commit']=git['research_commit']
        plan=out/'plan.json'
        if plan.exists() and json.loads(plan.read_text())!=identity:
            raise ValueError('matrix source/config/data/weights changed; use a new root')
        if not plan.exists():write_json(plan,identity)
        state=dict(status='running',**git,plan_sha256=digest(identity),runs={},phase='training')
        status_path=out/'status.json'
        def status(**kw):
            state.update(kw,updated_unix=time.time())
            tmp=status_path.with_suffix('.tmp.json');write_json(tmp,state);tmp.replace(status_path)
        def run(cmd,log,phase,run_id):
            env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with log.open('ab') as f:
                process=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
                while process.poll() is None:
                    status(phase=phase,current_run=run_id,subprocess_pid=process.pid)
                    time.sleep(15)
                return process.returncode
        try:
            # Freeze every training artifact before looking at final target outputs.
            for entry in matrix['runs']:
                run_id=entry['variant']+'-'+str(entry['seed'])
                directory=out/'training'/run_id;directory.mkdir(parents=True,exist_ok=True)
                attempts=sorted(directory.glob('attempt-*'))
                record=None
                if attempts:
                    previous=attempts[-1]
                    p=previous/'provenance.json'
                    record=json.loads(p.read_text()) if p.exists() else {}
                    if record.get('status')=='complete':
                        if (record['config']!=configs[entry['variant']] or record['seed']!=entry['seed'] or
                            sha256(previous/'uap.npy')!=record['uap_sha256']):
                            raise ValueError('completed training identity mismatch')
                        state['runs'][run_id]=dict(status='complete',training=str(previous));status();continue
                    if record.get('status')=='failed':
                        state['runs'][run_id]=dict(status='failed',training=str(previous),reason=record.get('error_reason'));status();continue
                target=directory/f'attempt-{len(attempts):03d}'
                cmd=[sys.executable,'tools/train.py','--config',entry['config'],'--seed',str(entry['seed']),
                    '--manifest-dir',a.manifest_dir,'--baseline-record',a.baseline_record,
                    '--weights-lock',a.weights_lock,'--weights-dir',a.weights_dir,
                    '--determinism','warn','--device',a.device,'--output',str(target)]
                checkpoints=sorted(directory.glob('attempt-*/checkpoints/*.pt'))
                if checkpoints:cmd+=['--resume',str(checkpoints[-1])]
                code=run(cmd,directory/'execution.log','training',run_id)
                p=target/'provenance.json';record=json.loads(p.read_text()) if p.exists() else {}
                success=code==0 and record.get('status')=='complete' and record.get('step')==10000
                state['runs'][run_id]=dict(status='complete' if success else 'failed',training=str(target),
                    reason=None if success else record.get('error_reason',f'exit{code}'))
                status()
            for run_id,item in state['runs'].items():
                if item['status']!='complete':continue
                train=Path(item['training'])
                for name in PILOT_TARGETS:
                    destination=out/'evaluations'/run_id/name
                    metric=destination/name/'metrics.json'
                    if metric.exists():
                        try:
                            r=validate_metric(json.loads(metric.read_text()))
                            expected=json.loads((train/'provenance.json').read_text())
                            if r['uap_sha256']!=expected['uap_sha256'] or r['manifest_sha256']!=index['manifest_hashes']['holdout']:
                                raise ValueError('existing evaluation identity mismatch')
                            continue
                        except (KeyError,ValueError,TypeError):pass
                    if destination.exists():
                        archive=out/'quarantine'/run_id;archive.mkdir(parents=True,exist_ok=True)
                        destination.rename(archive/(name+'-'+str(time.time_ns())))
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    cmd=[sys.executable,'tools/evaluate.py','--uap',str(train/'uap.npy'),
                        '--training-record',str(train/'provenance.json'),'--data-root',a.data_root,
                        '--manifest',str(Path(a.manifest_dir)/'holdout.csv'),'--targets',name,
                        '--weights-lock',a.weights_lock,'--weights-dir',a.weights_dir,
                        '--stage','final','--device',a.device,'--batch-size','32','--workers','4',
                        '--output',str(destination)]
                    code=run(cmd,destination.parent/(name+'.log'),'evaluation',run_id+'/'+name)
                    if code:state.setdefault('evaluation_failures',[]).append(run_id+'/'+name)
                    status()
            summary=out/'summary'
            if summary.exists():summary.rename(out/('summary-previous-'+str(time.time_ns())))
            result=aggregate(out/'evaluations',summary)
            status(status=result['status'],phase='finished',aggregate=result,subprocess_pid=None)
            print(json.dumps(state,indent=2))
            if result['status']!='complete':raise SystemExit(1)
        except BaseException as e:
            if state.get('phase')!='finished':status(status='interrupted',error_reason=type(e).__name__+': '+str(e))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',default='configs/run_matrix.json')
    p.add_argument('--manifest-dir',default='manifests/imagenet_val_devkit_v1')
    p.add_argument('--baseline-record',required=True)
    p.add_argument('--weights-lock',default='configs/targets/weights.lock.json')
    p.add_argument('--weights-dir',default='models_cache')
    p.add_argument('--data-root',default='data/imagenet2012/val')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',required=True)
    p.add_argument('--dry-run',action='store_true')
    execute(p.parse_args())
