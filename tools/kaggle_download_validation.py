"""Acquire only the contiguous validation-image byte span from the authorized Kaggle ZIP.
Run with the separate environments/kaggle.txt environment; no credential values are logged.
"""
import os
os.umask(0o077)
import concurrent.futures,hashlib,json,struct,time,zlib
from pathlib import Path
import requests,zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest
import _bootstrap
from kaggle_inspect_zip import RangeReader


def main():
    out=Path('outputs/kaggle-api');record=json.loads((out/'competition-zip-validation-index.json').read_text())
    assert record['validation_count']==50000
    assert record['contiguous_span_bytes']<=record['validation_compressed_bytes']+20*1024*1024
    assert record['contiguous_span_bytes']<8*1024**3
    api=KaggleApi();api.authenticate()
    with api.build_kaggle_client() as c:
     req=ApiDownloadDataFilesRequest();req.competition_name=record['competition']
     response=c.competitions.competition_api_client.download_data_files(req)
     if response.status_code!=200:raise RuntimeError('Archive access failed')
     url=response.url;length=int(response.headers['Content-Length']);etag=response.headers.get('ETag');response.close()
    if length!=record['zip_total_bytes'] or etag!=record['zip_etag']:raise RuntimeError('Archive version changed')
    
    partsdir=Path('data/imagenet2012/kaggle_validation_parts');partsdir.mkdir(exist_ok=True)
    chunk_size=512*1024*1024
    span=record['contiguous_span_bytes'];begin=record['range_begin']
    specs=[dict(index=i,start=begin+offset,end_exclusive=begin+min(offset+chunk_size,span)) for i,offset in enumerate(range(0,span,chunk_size))]
    start_time=time.monotonic()
    
    def fetch(spec):
     target=partsdir/f"part-{spec['index']:03d}.bin";partial=target.with_suffix('.partial')
     expected_size=spec['end_exclusive']-spec['start']
     if target.exists() and target.stat().st_size==expected_size:
      h=hashlib.sha256()
      with target.open('rb') as f:
       for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
      return dict(spec,path=str(target),size=expected_size,sha256=h.hexdigest(),local_part_verified=True)
     for attempt in range(3):
      received=partial.stat().st_size if partial.exists() else 0
      if received>expected_size:raise ValueError('Partial chunk exceeds expected range')
      if received==expected_size:partial.replace(target);return fetch(spec)
      first=spec['start']+received;last=spec['end_exclusive']-1
      try:
       with requests.get(url,headers={'Range':f'bytes={first}-{last}','Accept-Encoding':'identity'},stream=True,timeout=(20,120)) as r:
        if r.status_code!=206:raise RuntimeError(f'Range request HTTP {r.status_code}')
        if r.headers.get('Content-Range')!=f'bytes {first}-{last}/{length}' or r.headers.get('ETag')!=etag:raise ValueError('Range identity mismatch')
        with partial.open('ab') as f:
         for block in r.iter_content(chunk_size=1024*1024):
          if block:f.write(block)
       if partial.stat().st_size!=expected_size:raise ValueError('Truncated range')
       partial.replace(target)
       return fetch(spec)
      except (requests.RequestException,RuntimeError) as e:
       if attempt==2:raise RuntimeError(f"Chunk {spec['index']} failed: {type(e).__name__}") from None
       time.sleep(2**attempt)
    
    results=[];last_report=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
     futures={pool.submit(fetch,s):s for s in specs}
     while futures:
      done,_=concurrent.futures.wait(futures,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
      for f in done:
       result=f.result();results.append(result);futures.pop(f)
       print('CHUNK_COMPLETE',result['index'],result['size'],flush=True)
      total=0
      for path in [*partsdir.glob('part-*.bin'),*partsdir.glob('part-*.partial')]:
       try:total+=path.stat().st_size
       except FileNotFoundError:pass
      elapsed=time.monotonic()-start_time
      progress=dict(status='downloading' if futures else 'download_complete',received_bytes=total,expected_bytes=span,completed_chunks=len(results),total_chunks=len(specs),elapsed_seconds=elapsed)
      (out/'validation-download-progress.json').write_text(json.dumps(progress,indent=2)+'\n')
      if elapsed-last_report>=20 or not futures:
       print('PROGRESS',json.dumps(progress),flush=True);last_report=elapsed
    results.sort(key=lambda x:x['index'])
    (out/'validation-chunks.json').write_text(json.dumps(dict(status='download_complete',zip_etag=etag,range_begin=begin,range_end_exclusive=record['range_end_exclusive'],chunks=results),indent=2)+'\n')
    print('VALIDATION_BYTES_COMPLETE',span,flush=True)

if __name__=='__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
