"""Verify original devkit MD5 and crosscheck every existing label/manifest entry."""
import _bootstrap
import argparse
import json
from psp_research.devkit_verification import verify_devkit_mapping


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devkit', required=True)
    p.add_argument('--labels-dir', required=True, help='existing label bundle to crosscheck')
    p.add_argument('--manifest-dir', required=True, help='existing immutable manifest bundle')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(verify_devkit_mapping(a.devkit, a.labels_dir, a.manifest_dir, a.output), indent=2))
