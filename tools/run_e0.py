"""Run unchanged official ResNet152 in the declared legacy core, then verify outputs.

The source checkout remains untouched. Re-evaluation includes all50000 validation
images; paper five-run means are contextual references, never stopping targets.
"""
import _bootstrap
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from torchvision import models
from psp_research.provenance import ROOT,UPSTREAM,sha256,write_json,check_upstream
from psp_research.registry import E0_NAMES,TARGETS,E0_TARGETS
from psp_research.metrics import load_uap


def execute(a):
    check_upstream()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if not out.is_relative_to(ROOT):raise ValueError('E0 output must be inside the project for relative provenance')
    source=out/'source'
    if not source.exists():
        subprocess.run(['git','worktree','add','--detach',str(source),UPSTREAM],check=True,cwd=ROOT)
    # Compare bytes directly with fixed upstream blobs, independently of mutable refs.
    import hashlib
    for name in ['LICENSE','README.md','main.py','PSP_UAP.py','loss.py','functions.py','semantic_aug.py','strategy.py']:
        original=subprocess.check_output(['git','show',UPSTREAM+':'+name],cwd=ROOT)
        if hashlib.sha256(original).hexdigest()!=sha256(source/name):raise ValueError('upstream checkout changed: '+name)
    checkpoints=source/'TorchHub/checkpoints';checkpoints.mkdir(parents=True,exist_ok=True)
    native_status=out/'native.status.json'
    if not native_status.exists():
        image_identity=subprocess.check_output(['docker','image','inspect',a.image,'--format','{{.Id}}'],text=True).strip()
        cmd=['docker','run','--rm','--gpus','all','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
            '--shm-size','4g','--mount',f'type=bind,src={Path(a.legacy_prefix).resolve()},dst=/opt/psp-legacy,readonly',
            '--mount',f'type=bind,src={source},dst=/work',
            '--mount',f'type=bind,src={Path(a.data_root).resolve()},dst=/data,readonly',
            '--mount',f'type=bind,src={Path(a.weights_dir).resolve()},dst=/work/TorchHub/checkpoints,readonly',
            image_identity,'-u','main.py','--surrogate_model','resnet152','--data_path','/data',
            '--p_rate','0.65','--p_active','--prior','jigsaw','--re_weight','--temper','3',
            '--input_transform','--seed','126','--device','cuda:0']
        with (out/'native.log').open('wb') as f:
            start=time.perf_counter()
            code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT).returncode
        write_json(native_status,dict(returncode=code,image=a.image,image_identity=image_identity,
            elapsed_seconds=time.perf_counter()-start,upstream_commit=UPSTREAM))
    native=json.loads(native_status.read_text())
    if native['returncode']!=0:raise RuntimeError('original legacy run failed; see native.log')
    found=list(source.glob('results/**/*.npy'))
    if len(found)!=1:raise ValueError('expected exactly one original returned UAP')
    uap=found[0];load_uap(uap)
    evaluation=out/'reloaded'
    if not (evaluation/'provenance.json').exists():
        cmd=[sys.executable,'tools/evaluate_e0.py','--uap',str(uap),'--data-root',a.data_root,
            '--manifest-dir',a.manifest_dir,'--weights-lock',a.weights_lock,
            '--weights-dir',a.weights_dir,'--execution-label','upstream_legacy_core',
            '--output',str(evaluation)]
        with (out/'reloaded.log').open('wb') as f:
            subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
    report=json.loads((evaluation/'provenance.json').read_text())
    if report['status']!='complete':raise ValueError('independent E0 evaluation incomplete')
    log=(out/'native.log').read_text(errors='replace')
    native_fr=dict((name,float(value)) for name,value in re.findall(r'on \((\w+)\) is ([\d.eE+-]+)',log))
    # main.py prints final FR as a percentage; its text log stores ratios separately.
    if set(native_fr)!=set(E0_NAMES):raise ValueError('original five-model final evaluation missing')
    discrepancies={}
    paper=[58.82,88.59,87.35,85.65,76.00]
    for name,paper_fr in zip(E0_NAMES,paper):
        r=report['targets'][name]
        enum,member,_=(TARGETS|E0_TARGETS)[name]
        expected_clean=getattr(getattr(models,enum),member).meta['_metrics']['ImageNet-1K']['acc@1']/100
        native_difference=abs(r['fr']-native_fr[name]/100)
        clean_difference=abs(r['clean_acc']-expected_clean)
        if native_difference>.002 or clean_difference>.02:
            raise ValueError('baseline preprocessing/metrics discrepancy: '+name)
        discrepancies[name]=dict(native_fr_percent=native_fr[name],reloaded_fr=r['fr'],
            native_difference=native_difference,reference_clean_acc=expected_clean,
            clean_difference=clean_difference,paper_five_run_mean_fr_percent=paper_fr,
            single_seed_minus_paper_mean_pp=r['fr']*100-paper_fr)
    report.update(baseline_checks_passed=True,uap_path=uap.relative_to(ROOT).as_posix(),
        native_log_sha256=sha256(out/'native.log'),native_source_commit=UPSTREAM,
        baseline_checks=discrepancies,paper_comparison='Single seed; does not establish reproduction of paper five-run mean/std',
        paper_reference='https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Data-free_Universal_Adversarial_Perturbation_with_Pseudo-semantic_Prior_CVPR_2025_paper.pdf',
        native_numpy_seeded=False,original_bugs_preserved=True)
    write_json(a.baseline_record,report)
    print('E0_ENGINEERING_CHECKS_PASS',a.baseline_record,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--legacy-prefix',required=True)
    p.add_argument('--image',default='psp-uap-legacy:core-20261126')
    p.add_argument('--data-root',default='data/imagenet2012/val')
    p.add_argument('--manifest-dir',default='manifests/imagenet_val_devkit_v1')
    p.add_argument('--weights-lock',default='configs/targets/e0_weights.lock.json')
    p.add_argument('--weights-dir',default='models_cache')
    p.add_argument('--baseline-record',required=True)
    execute(p.parse_args())
