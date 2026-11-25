"""Build devkit-verified ImageNet validation manifests without copying images."""
import _bootstrap
import argparse
from psp_research.data import make_manifests, devkit_labels

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('devkit', help='extract authoritative mapping from licensed official devkit')
    d.add_argument('--devkit', required=True)
    d.add_argument('--out', required=True)
    m = sub.add_parser('split', help='verify all filenames/labels/content and freeze 1000/49000 split')
    m.add_argument('--data-root', required=True)
    m.add_argument('--labels-dir', required=True)
    m.add_argument('--out', required=True)
    m.add_argument('--seed', type=int, choices=[126], default=126)
    a = p.parse_args()
    if a.command == 'devkit':
        devkit_labels(a.devkit, a.out)
    else:
        make_manifests(a.data_root, a.labels_dir, a.out, a.seed)
