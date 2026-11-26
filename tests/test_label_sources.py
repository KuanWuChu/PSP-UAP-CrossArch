import csv
import json
import pytest
from psp_research.label_sources import kaggle_labels
from psp_research.data import make_manifests


@pytest.fixture
def inputs(tmp_path):
    classes=[f'n{i:08d}' for i in range(1,1001)]
    labels=tmp_path/'labels.txt'; labels.write_text('\n'.join(classes)+'\n')
    truth=[classes[i%1000] for i in range(50000)]
    reference=tmp_path/'truth.txt';reference.write_text('\n'.join(truth)+'\n')
    csv_file=tmp_path/'labels.csv'
    with csv_file.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['ImageId','PredictionString'])
        # Row order must not determine the mapping.
        for i in reversed(range(50000)):
            writer.writerow([f'ILSVRC2012_val_{i+1:08d}',f'{truth[i]} 0 0 10 10 {truth[i]} 1 1 9 9'])
    return csv_file,labels,reference


def test_crosschecked_source_is_explicitly_not_a_verified_devkit(inputs,tmp_path):
    source=kaggle_labels(*inputs,'a'*40,tmp_path/'out')
    assert source['official_devkit_verified'] is False and source['devkit_sha256'] is None
    assert source['matched_validation_labels']==50000
    assert json.loads((tmp_path/'out/class_index.json').read_text())['999'][0]=='n00001000'
    assert (tmp_path/'out/validation_wnids.txt').read_bytes()==inputs[2].read_bytes()
    with pytest.raises(FileExistsError):
        kaggle_labels(*inputs,'a'*40,tmp_path/'out')
    # Hash validation applies equally to both explicitly labeled provenance routes.
    (tmp_path/'out/class_index.json').write_text('{}')
    with pytest.raises(ValueError,match='digest'):
        make_manifests(tmp_path,tmp_path/'out',tmp_path/'manifests')


@pytest.mark.parametrize('case',['wrong_label','duplicate','missing','ambiguous','invalid_id','bad_commit'])
def test_invalid_source_rejected(inputs,tmp_path,case):
    csv_file,labels,reference=inputs
    rows=list(csv.reader(csv_file.open()))
    if case=='wrong_label':rows[1][1]='n00000001 0 0 10 10'
    elif case=='duplicate':rows.append(rows[1])
    elif case=='missing':rows.pop()
    elif case=='ambiguous':rows[1][1]+=' n00000001 0 0 10 10'
    elif case=='invalid_id':rows[1][0]='../../invalid'
    with csv_file.open('w',newline='') as f:csv.writer(f).writerows(rows)
    with pytest.raises(ValueError):
        kaggle_labels(csv_file,labels,reference,'main' if case=='bad_commit' else 'a'*40,tmp_path/'out')
