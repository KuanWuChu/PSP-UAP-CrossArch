"""Verify SHA256/CRC32 and JPEG integrity, then extract the selected validation range."""
import _bootstrap
import argparse,json
from psp_research.validation_archive import extract_validation

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',required=True)
    p.add_argument('--chunks',required=True)
    p.add_argument('--parts-dir',required=True)
    p.add_argument('--labels-dir',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--inventory',required=True)
    a=p.parse_args()
    print(json.dumps(extract_validation(a.index,a.chunks,a.parts_dir,a.labels_dir,a.data_root,a.inventory),indent=2))
