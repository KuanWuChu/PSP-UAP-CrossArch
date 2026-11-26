import copy
import pytest
from psp_research.execution import require_baseline
from psp_research.provenance import write_json,sha256,UPSTREAM
from psp_research.registry import E0_NAMES
from psp_research.aggregate import paired_delta
from test_aggregate import record


@pytest.fixture
def baseline(tmp_path):
    uap=tmp_path/'fixture-artifact';uap.write_bytes(b'identity-only fixture, not an experiment UAP')
    manifests={'selection':'1'*64,'holdout':'2'*64,'smoke_holdout':'3'*64}
    r=dict(status='complete',upstream_commit=UPSTREAM,baseline_checks_passed=True,
        execution_label='upstream_modern_compatibility',manifest_hashes=manifests,
        uap_path=str(uap),uap_sha256=sha256(uap),targets={name:dict(total_count=50000,
        flip_count=25000,fr=.5,clean_correct_count=40000,clean_acc=.8) for name in E0_NAMES})
    p=tmp_path/'baseline.json';write_json(p,r)
    return p,r,manifests


def test_missing_or_unreviewed_e0_blocks_full_training(baseline):
    p,r,m=baseline
    with pytest.raises(ValueError,match='requires'):
        require_baseline(None,m)
    r['baseline_checks_passed']=False;write_json(p,r)
    with pytest.raises(ValueError,match='not complete'):
        require_baseline(p,m)


def test_baseline_identity_and_counts_cannot_be_replaced(baseline):
    p,r,m=baseline
    assert require_baseline(p,m)['baseline_uap_sha256']==r['uap_sha256']
    for change,reason in [(dict(targets={}), 'five CNN'),
                          (dict(manifest_hashes={}), 'data validation')]:
        write_json(p,dict(r,**change))
        with pytest.raises(ValueError,match=reason):require_baseline(p,m)
    wrong=copy.deepcopy(r);wrong['targets']['alexnet']['flip_count']=25001
    write_json(p,wrong)
    with pytest.raises(ValueError,match='counts'):require_baseline(p,m)
    write_json(p,r);p.parent.joinpath('fixture-artifact').write_bytes(b'changed')
    with pytest.raises(ValueError,match='content changed'):require_baseline(p,m)


def test_comparisons_reject_different_training_numerical_modes():
    a,b=record(),record('e2_psp_only')
    a['training_numerical_policy']={'deterministic_warn_only':False}
    b['training_numerical_policy']={'deterministic_warn_only':True}
    with pytest.raises(ValueError,match='identities'):paired_delta(a,b,'e2')


def test_incomplete_matrix_cannot_export_final_results(tmp_path):
    import subprocess,sys
    matrix=tmp_path/'matrix';matrix.mkdir()
    write_json(matrix/'status.json',dict(status='incomplete'))
    write_json(tmp_path/'baseline.json',dict(baseline_checks_passed=False))
    out=tmp_path/'public';artifact=tmp_path/'uaps.zip'
    result=subprocess.run([sys.executable,'tools/package_results.py','--matrix',str(matrix),
        '--baseline-record',str(tmp_path/'baseline.json'),'--execution-root',str(tmp_path),
        '--output',str(out),'--artifact',str(artifact)],capture_output=True,text=True)
    assert result.returncode!=0 and 'refusing partial final package' in result.stderr
    assert not out.exists() and not artifact.exists()
