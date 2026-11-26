import io,struct,zlib
from pathlib import Path
from PIL import Image
import pytest
from psp_research.validation_archive import SegmentedReader,read_validation_entry,PREFIX


def make_entry(tmp_path, zip64=False):
    image=io.BytesIO();Image.new('RGB',(8,8),(30,80,130)).save(image,format='JPEG');data=image.getvalue()
    compressor=zlib.compressobj(wbits=-15);payload=compressor.compress(data)+compressor.flush()
    name=PREFIX+'ILSVRC2012_val_00000001.JPEG';encoded=name.encode()
    entry=dict(name=name,offset=100,compression=8,flags=0,crc32=zlib.crc32(data),compressed_size=len(payload),size=len(data))
    extra=struct.pack('<HHQQ',1,16,len(data),len(payload)) if zip64 else b''
    header=struct.pack('<4s5H3I2H',b'PK\x03\x04',45 if zip64 else 20,0,8,0,0,entry['crc32'],0xffffffff if zip64 else len(payload),0xffffffff if zip64 else len(data),len(encoded),len(extra))
    blob=header+encoded+extra+payload
    paths=[];sizes=[]
    for i,part in enumerate([blob[:31],blob[31:100],blob[100:]]):
        p=tmp_path/f'part{i}';p.write_bytes(part);paths.append(p);sizes.append(len(part))
    return entry,data,SegmentedReader(paths,sizes)


@pytest.mark.parametrize('zip64',[False,True])
def test_cross_segment_header_payload_and_crc(tmp_path,zip64):
    entry,expected,reader=make_entry(tmp_path,zip64)
    try:
        assert read_validation_entry(reader,entry,100)==(1,expected)
        assert reader.pos==reader.size
        reader.seek(reader.size);assert reader.read(10)==b''
        with pytest.raises(ValueError):reader.seek(-1)
        with pytest.raises(ValueError):reader.seek(reader.size+1)
    finally:reader.close()


@pytest.mark.parametrize('change',[dict(name='../unsafe.JPEG'),dict(crc32=0),dict(size=1),dict(compression=0)])
def test_reject_path_or_catalog_tampering(tmp_path,change):
    entry,_,reader=make_entry(tmp_path)
    try:
        with pytest.raises(ValueError):read_validation_entry(reader,dict(entry,**change),100)
    finally:reader.close()


def test_zip64_missing_size_extra():
    from psp_research.zip_utils import local_sizes
    with pytest.raises(ValueError,match='missing ZIP64'):
        local_sizes(0xffffffff,0xffffffff,b'')


def test_corrupt_compressed_payload_rejected(tmp_path):
    entry,_,reader=make_entry(tmp_path,True)
    reader.close()
    path=tmp_path/'part2';data=bytearray(path.read_bytes());data[-1]^=0xff;path.write_bytes(data)
    reader=SegmentedReader([tmp_path/'part0',tmp_path/'part1',path],[(tmp_path/'part0').stat().st_size,(tmp_path/'part1').stat().st_size,path.stat().st_size])
    try:
        with pytest.raises((ValueError,zlib.error)):
            read_validation_entry(reader,entry,100)
    finally:reader.close()
