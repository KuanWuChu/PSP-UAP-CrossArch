"""Package only a complete prespecified matrix, with raw counts and standard plots.

No images, classification weights, credentials or private logs are exported.
The UAP ZIP is a local/versioned release artifact, never a Git-tracked dataset.
"""
import _bootstrap
import argparse
import json
import csv
import shutil
import zipfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from psp_research.aggregate import aggregate,validate_metric,VARIANTS,SEEDS,PILOT_TARGETS
from psp_research.metrics import load_uap
from psp_research.provenance import sha256,write_json,UPSTREAM


def execute(a):
    matrix=Path(a.matrix);state=json.loads((matrix/'status.json').read_text())
    baseline=json.loads(Path(a.baseline_record).read_text())
    if state['status']!='complete' or baseline.get('baseline_checks_passed') is not True:
        raise ValueError('complete matrix and verified baseline required; refusing partial final package')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    result=aggregate(matrix/'evaluations',out/'summary')
    if result['status']!='complete':raise ValueError('independent aggregation is incomplete')
    artifacts={};training={};metrics=[]
    for variant in VARIANTS:
        for seed in SEEDS:
            run_id=variant+'-'+str(seed)
            path=Path(state['runs'][run_id]['training'])
            if not path.is_absolute():path=Path(a.execution_root)/path
            p=json.loads((path/'provenance.json').read_text())
            if (p['status']!='complete' or p['purpose']!='full' or p['step']!=10000 or
                p['config']['variant']!=variant or p['seed']!=seed or p['upstream_commit']!=UPSTREAM):
                raise ValueError('training provenance mismatch: '+run_id)
            load_uap(path/'uap.npy')
            if sha256(path/'uap.npy')!=p['uap_sha256']:raise ValueError('UAP hash changed')
            artifacts['uaps/'+run_id+'.npy']=path/'uap.npy'
            training[run_id]=p
    e0_uap=Path(baseline['uap_path']);load_uap(e0_uap)
    if sha256(e0_uap)!=baseline['uap_sha256']:raise ValueError('E0 UAP hash changed')
    artifacts['uaps/e0-resnet152-126.npy']=e0_uap
    for path in sorted((matrix/'evaluations').rglob('metrics.json')):
        r=validate_metric(json.loads(path.read_text()))
        run_id=r['variant']+'-'+str(r['seed'])
        if r['uap_sha256']!=training[run_id]['uap_sha256']:raise ValueError('evaluated UAP differs from canonical run')
        metrics.append(r)
    if len(metrics)!=90:raise ValueError('expected90 unambiguous target records')
    public_baseline={k:v for k,v in baseline.items() if k not in {'uap_path'}}
    public_baseline['uap_artifact']='uaps/e0-resnet152-126.npy'
    write_json(out/'e0_baseline.json',public_baseline)
    write_json(out/'target_metrics.json',metrics)
    write_json(out/'training_provenance.json',training)
    plan=json.loads((matrix/'plan.json').read_text())
    write_json(out/'experiment_identity.json',dict(source_commit=state['research_commit'],
        upstream_commit=UPSTREAM,plan=plan,aggregate=result,
        scope='Modern fixed-budget research; independent legacy-core single-seed E0; not a pure architecture-only intervention',
        no_target_checkpoint_selection=True,original_loss_unchanged=True))
    def read(path):
        with path.open() as f:return list(csv.DictReader(f))
    means=read(out/'summary/seed_summary.csv');deltas=read(out/'summary/paired_delta_summary.csv')
    def metric(v,t,m):return next(r for r in means if r['variant']==v and r['target']==t and r['metric']==m)
    def delta(v,t):return next(r for r in deltas if r['intervention']==v and r['target']==t)
    def fmt(r,scale=1):return f"{float(r['mean'])*scale:.2f} ± {float(r['sample_std'])*scale:.2f}"
    fig,axes=plt.subplots(3,1,figsize=(10,12),constrained_layout=True)
    x=np.arange(len(PILOT_TARGETS))
    for ax in axes:
        ax.set_xticks(x,PILOT_TARGETS,rotation=20,ha='right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    rr=[metric('full_original',t,'fr') for t in PILOT_TARGETS]
    axes[0].bar(x,[float(r['mean'])*100 for r in rr],yerr=[float(r['sample_std'])*100 for r in rr],capsize=4)
    axes[0].set(title='E1: same CNN-trained UAP per seed',ylabel='FR (%)')
    rr=[delta('e2_psp_only',t) for t in PILOT_TARGETS]
    axes[1].bar(x,[float(r['mean']) for r in rr],yerr=[float(r['sample_std']) for r in rr],capsize=4)
    axes[1].axhline(0,color='black',linewidth=.8);axes[1].set(title='E2: PSP-only B minus crop-matched noise A',ylabel='ΔFR (percentage points)')
    for i,v in enumerate(['full_local','full_large']):
        rr=[delta(v,t) for t in PILOT_TARGETS]
        axes[2].bar(x+(i-.5)*.35,[float(r['mean']) for r in rr],.35,yerr=[float(r['sample_std']) for r in rr],capsize=3,label=v+' minus original')
    axes[2].axhline(0,color='black',linewidth=.8);axes[2].set(title='E3: prespecified crop policies',ylabel='ΔFR (percentage points)');axes[2].legend()
    fig.savefig(out/'comparison.png',dpi=200);fig.savefig(out/'comparison.svg',metadata={'Date':None});plt.close(fig)
    lines=['# Complete prespecified PSP-UAP research matrix','',
        '15 fixed-budget runs ×10,000 steps;90 target evaluations ×49,000 fixed holdout images. Three seeds126/2025/3407; sample standard deviation(ddof1), not confidence intervals. All training artifacts frozen before final evaluation.','',
        'E0 used unchanged upstream source in the declared legacy core; research runs use the documented modern compatibility loop and explicit CUDA warn-only mode. GPU bitwise replay is not claimed. Different target pretraining recipes/preprocessing prevent a pure architecture-only causal interpretation.','',
        '![FR and paired changes](comparison.png)','',
        '| Target | Full PSP FR (%) | Clean accuracy (%) | Adv accuracy (%) | Clean-correct ASR (%) |','|---|---:|---:|---:|---:|']
    for t in PILOT_TARGETS:lines.append('|'+t+'|'+'|'.join(fmt(metric('full_original',t,m),100) for m in ['fr','clean_acc','adv_acc','cc_asr'])+'|')
    lines+=['','| Target | E2 B−A (pp) | E3 local−original (pp) | E3 large−original (pp) |','|---|---:|---:|---:|']
    for t in PILOT_TARGETS:lines.append('|'+t+'|'+'|'.join(fmt(delta(v,t)) for v in ['e2_psp_only','full_local','full_large'])+'|')
    lines+=['','Raw integer counts are in `summary/seed_metrics.csv`; all full identities are in `target_metrics.json` and `training_provenance.json`. The original supplement already includes DeiT3; its exact layer selection remains unresolved. No new loss or classifier training was introduced.','',
        'E0 is one original-source run, compared separately with the paper five-run mean; it does not establish reproduction of the paper mean/std. See `e0_baseline.json`.','']
    (out/'README.md').write_text('\n'.join(lines))
    # Export whitelist only. Metadata must contain no machine-specific paths or tokens.
    for path in out.rglob('*'):
        if path.suffix in {'.json','.md','.csv','.svg'}:
            text=path.read_text()
            if any(s in text for s in ['/home/','X-Goog-Signature','github_pat_','ghp_']):
                raise ValueError('private path or credential pattern in public result: '+path.name)
    artifact_path=Path(a.artifact);artifact_path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(artifact_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(artifacts.items()):z.write(path,name)
        for path in sorted(out.rglob('*')):
            if path.is_file():z.write(path,'reports/'+path.relative_to(out).as_posix())
        z.write('LICENSE','UPSTREAM_MIT_LICENSE.txt')
    write_json(out/'uap_artifact.json',dict(file_name=artifact_path.name,sha256=sha256(artifact_path),
        bytes=artifact_path.stat().st_size,uaps={name:sha256(p) for name,p in artifacts.items()}))
    print('COMPLETE_RESULT_PACKAGE',out,artifact_path,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',required=True)
    p.add_argument('--execution-root',required=True)
    p.add_argument('--baseline-record',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--artifact',required=True)
    execute(p.parse_args())
