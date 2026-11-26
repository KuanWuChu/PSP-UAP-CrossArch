"""Inspect the authenticated competition ZIP with verified HTTP byte ranges.
Signed URLs and API credentials stay in process memory and are never printed.
"""
import io,json,re,sys,zipfile,time
import _bootstrap
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest

class RangeReader(io.RawIOBase):
    def __init__(self,url,length,etag):
        self.url,self.length,self.etag=url,length,etag
        self.pos=0;self.bytes_read=0;self.session=requests.Session();self.calls=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.length+offset
        if self.pos<0:raise ValueError('Negative seek')
        return self.pos
    def read(self,size=-1):
        if self.pos>=self.length or size==0:return b''
        if size<0:size=self.length-self.pos
        size=min(size,self.length-self.pos)
        if size>768*1024*1024:raise ValueError('Oversized metadata read refused')
        start,end=self.pos,self.pos+size-1
        headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}
        with self.session.get(self.url,headers=headers,stream=True,timeout=(20,180)) as r:
            if r.status_code!=206:
                raise RuntimeError(f'Expected HTTP206 byte range, received {r.status_code}; full response refused')
            expected=f'bytes {start}-{end}/{self.length}'
            if r.headers.get('Content-Range')!=expected:raise ValueError('Content-Range mismatch')
            if self.etag and r.headers.get('ETag')!=self.etag:raise ValueError('Remote archive identity changed')
            data=r.content
        if len(data)!=size:raise ValueError('Truncated range response')
        self.pos+=size;self.bytes_read+=size;self.calls+=1
        return data

if __name__=='__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    out=Path('outputs/kaggle-api')
    api=KaggleApi();api.authenticate()
    with api.build_kaggle_client() as c:
        req=ApiDownloadDataFilesRequest();req.competition_name='imagenet-object-localization-challenge'
        response=c.competitions.competition_api_client.download_data_files(req)
        if response.status_code!=200:raise RuntimeError('Archive metadata request failed')
        url=response.url;length=int(response.headers['Content-Length']);etag=response.headers.get('ETag')
        response.close()
    print('ARCHIVE_HEADERS',json.dumps(dict(total_bytes=length,has_etag=bool(etag))),flush=True)
    reader=RangeReader(url,length,etag)
    start=time.monotonic()
    with zipfile.ZipFile(reader) as z:
        entries=z.infolist()
        selected=[x for x in entries if re.fullmatch(r'ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_\d{8}\.JPEG',x.filename)]
        selected.sort(key=lambda x:x.header_offset)
        assert len(selected)==50000 and len({x.filename for x in selected})==50000
        ordered=sorted(entries,key=lambda x:x.header_offset)
        offsets=[x.header_offset for x in ordered]
        import bisect
        begin=selected[0].header_offset
        next_index=bisect.bisect_right(offsets,selected[-1].header_offset)
        end=offsets[next_index] if next_index<len(offsets) else z.start_dir
        compressed=sum(x.compress_size for x in selected)
        inside=[x.filename for x in ordered if begin <= x.header_offset < end]
        if len(inside)!=50000 or set(inside)!={x.filename for x in selected}:
            raise ValueError('validation entries are not an exclusive contiguous span')
        record=dict(competition='imagenet-object-localization-challenge',zip_total_bytes=length,zip_etag=etag,
            total_entries=len(entries),validation_count=len(selected),validation_uncompressed_bytes=sum(x.file_size for x in selected),
            validation_compressed_bytes=compressed,range_begin=begin,range_end_exclusive=end,
            contiguous_span_bytes=end-begin,metadata_http_bytes=reader.bytes_read,metadata_http_calls=reader.calls,
            candidate_devkits=[x.filename for x in entries if 'devkit' in x.filename.lower() or x.filename.endswith('/meta.mat')],
            mapping_files=[x.filename for x in entries if x.filename.endswith('LOC_synset_mapping.txt')],
            validation_entries=[dict(name=x.filename,offset=x.header_offset,compressed_size=x.compress_size,
                size=x.file_size,crc32=x.CRC,compression=x.compress_type,flags=x.flag_bits) for x in selected])
        (out/'competition-zip-validation-index.json').write_text(json.dumps(record,indent=2)+'\n')
        print('VALIDATION_INDEX',json.dumps({k:v for k,v in record.items() if k!='validation_entries'}),flush=True)
        print('INSPECTION_SECONDS',time.monotonic()-start,flush=True)
