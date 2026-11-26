"""Verify and extract only validation entries from authenticated ZIP byte ranges."""
import bisect
import csv
import hashlib
import io
import json
import re
import struct
import zlib
from pathlib import Path
from PIL import Image
from .provenance import sha256, write_json
from .zip_utils import local_sizes

PREFIX = 'ILSVRC/Data/CLS-LOC/val/'


class SegmentedReader:
    def __init__(self, paths, sizes):
        if not paths or len(paths) != len(sizes) or any(n <= 0 for n in sizes):
            raise ValueError('invalid segment catalog')
        self.files = []
        try:
            for p, n in zip(paths, sizes):
                if Path(p).stat().st_size != n:
                    raise ValueError('segment size mismatch')
                self.files.append(Path(p).open('rb'))
        except Exception:
            self.close()
            raise
        self.ends = []
        total = 0
        for n in sizes:
            total += n
            self.ends.append(total)
        self.pos, self.size = 0, total

    def close(self):
        for f in self.files:
            f.close()

    def seek(self, offset):
        if not 0 <= offset <= self.size:
            raise ValueError('seek outside selected validation range')
        self.pos = offset

    def read(self, n):
        if n < 0 or n > 64*1024*1024:
            raise ValueError('invalid validation entry read size')
        remaining = min(n, self.size - self.pos)
        blocks = []
        while remaining:
            i = bisect.bisect_right(self.ends, self.pos)
            previous = self.ends[i-1] if i else 0
            length = min(remaining, self.ends[i] - self.pos)
            self.files[i].seek(self.pos - previous)
            b = self.files[i].read(length)
            if len(b) != length:
                raise ValueError('truncated segment')
            blocks.append(b)
            self.pos += length
            remaining -= length
        return b''.join(blocks)


def read_validation_entry(reader, entry, range_begin):
    match = re.fullmatch(re.escape(PREFIX) + r'ILSVRC2012_val_(\d{8})\.JPEG', entry['name'])
    if not match or not 1 <= int(match[1]) <= 50000:
        raise ValueError('unsafe or non-validation archive entry')
    reader.seek(entry['offset'] - range_begin)
    raw = reader.read(30)
    if len(raw) != 30:
        raise ValueError('truncated local header')
    signature, version, flags, method, mtime, mdate, crc, csize, size, nlen, xlen = struct.unpack('<4s5H3I2H', raw)
    if signature != b'PK\x03\x04' or flags & 9 or method not in (0, 8):
        raise ValueError('unsupported or encrypted validation header')
    name = reader.read(nlen).decode('utf-8' if flags & 0x800 else 'cp437')
    extra = reader.read(xlen)
    csize, size = local_sizes(csize, size, extra)
    if (name != entry['name'] or method != entry['compression'] or flags != entry['flags'] or
        crc != entry['crc32'] or csize != entry['compressed_size'] or size != entry['size']):
        raise ValueError('local header disagrees with authenticated central directory')
    payload = reader.read(csize)
    data = zlib.decompress(payload, -15) if method == 8 else payload
    if len(data) != size or zlib.crc32(data) != crc:
        raise ValueError('validation image size/CRC32 mismatch')
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'JPEG':
            raise ValueError('expected native JPEG')
        image.verify()
    return int(match[1]), data


def extract_validation(index_file, chunks_file, parts_dir, labels_dir, data_root, inventory):
    index = json.loads(Path(index_file).read_text())
    chunks = json.loads(Path(chunks_file).read_text())
    labels_dir, data_root, parts_dir = Path(labels_dir), Path(data_root), Path(parts_dir).resolve()
    source = json.loads((labels_dir / 'source.json').read_text())
    if sha256(labels_dir/'validation_wnids.txt') != source['ground_truth_sha256']:
        raise ValueError('label digest mismatch')
    truth = (labels_dir/'validation_wnids.txt').read_text().splitlines()
    if len(truth) != 50000 or any(not re.fullmatch(r'n\d{8}', w) for w in truth):
        raise ValueError('invalid labels')
    if (chunks['status'] != 'download_complete' or chunks['zip_etag'] != index['zip_etag'] or
        chunks['range_begin'] != index['range_begin'] or chunks['range_end_exclusive'] != index['range_end_exclusive']):
        raise ValueError('download chunks do not match archive catalog')
    entries = index['validation_entries']
    ids = [int(e['name'][len(PREFIX+'ILSVRC2012_val_'):-5]) for e in entries if re.fullmatch(re.escape(PREFIX)+r'ILSVRC2012_val_\d{8}\.JPEG',e['name'])]
    if len(ids) != 50000 or set(ids) != set(range(1, 50001)) or len(entries) != 50000:
        raise ValueError('requires exactly the original 50000 validation IDs')
    paths, sizes = [], []
    position = index['range_begin']
    for i, part in enumerate(chunks['chunks']):
        path = parts_dir / f'part-{i:03d}.bin'
        if (part['index'] != i or part['start'] != position or part['size'] != part['end_exclusive']-part['start'] or
            not path.resolve().is_relative_to(parts_dir) or sha256(path) != part['sha256']):
            raise ValueError('chunk sequence/path/SHA256 mismatch')
        paths.append(path);sizes.append(part['size']);position=part['end_exclusive']
    if position != index['range_end_exclusive']:
        raise ValueError('incomplete selected archive span')
    inventory = Path(inventory)
    inventory.parent.mkdir(parents=True, exist_ok=True)
    temporary = inventory.with_suffix('.partial.csv')
    data_root.mkdir(parents=True, exist_ok=True)
    reader = SegmentedReader(paths, sizes)
    total_bytes = 0
    try:
        with temporary.open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=['image_id','class_wnid','bytes','source_crc32','file_sha256'],lineterminator='\n')
            writer.writeheader()
            for count, entry in enumerate(entries,1):
                number, data = read_validation_entry(reader, entry, index['range_begin'])
                expected_next = entries[count]['offset'] if count < len(entries) else index['range_end_exclusive']
                if reader.pos + index['range_begin'] != expected_next:
                    raise ValueError('selected span contains a gap or non-validation bytes')
                wnid = truth[number-1]
                relative = f'{wnid}/ILSVRC2012_val_{number:08d}.JPEG'
                target = data_root / relative
                target.parent.mkdir(exist_ok=True)
                image_hash = hashlib.sha256(data).hexdigest()
                if target.exists():
                    if sha256(target) != image_hash:
                        raise ValueError('existing image content disagrees with source')
                else:
                    pending = target.with_suffix('.partial')
                    pending.write_bytes(data)
                    pending.replace(target)
                writer.writerow(dict(image_id=relative,class_wnid=wnid,bytes=len(data),source_crc32=entry['crc32'],file_sha256=image_hash))
                total_bytes += len(data)
                if count % 5000 == 0:
                    f.flush();print(f'VALIDATED_AND_EXTRACTED {count}/50000',flush=True)
    finally:
        reader.close()
    temporary.replace(inventory)
    report = dict(status='complete',images=50000,classes=len(set(truth)),image_bytes=total_bytes,
        crc32_verified_images=50000,jpeg_verified_images=50000,image_inventory_sha256=sha256(inventory),
        archive_catalog_sha256=sha256(index_file),chunk_catalog_sha256=sha256(chunks_file),
        source_zip_etag=index['zip_etag'],original_validation_tar_md5_verified=False,
        label_source=source)
    write_json(inventory.with_suffix('.verification.json'),report)
    return report
