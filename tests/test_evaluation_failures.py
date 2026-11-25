from argparse import Namespace
import json
import pytest
from psp_research.evaluator import evaluate


def test_missing_dataset_leaves_failed_artifact(tmp_path):
    out = tmp_path / 'evaluation'
    a = Namespace(output=str(out), epsilon=10/255, batch_size=1, workers=0,
                  manifest=str(tmp_path / 'holdout.csv'))
    with pytest.raises(FileNotFoundError):
        evaluate(a)
    record = json.loads((out / 'provenance.json').read_text())
    assert record['status'] == 'failed' and record['error_reason'].startswith('FileNotFoundError')
    assert 'fr' not in record
    with pytest.raises(FileExistsError):
        evaluate(a)
