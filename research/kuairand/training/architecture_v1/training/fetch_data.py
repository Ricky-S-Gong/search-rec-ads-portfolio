"""Colab Drive SDK download; select TRAIN/validation data, never test targets."""
import hashlib
import io
import json
from pathlib import Path


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def fetch(service,root,release_path):
    from googleapiclient.http import MediaIoBaseDownload
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    release=json.loads(Path(release_path).read_text())
    files={x['relative']:x for x in release['files']}
    def download(name):
        record=files[name];path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Unsafe dataset path')
        if path.exists() and checksum(path)==record['sha256']:return
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.partial')
        request=service.files().get_media(fileId=record['id'],supportsAllDrives=True)
        with temporary.open('wb') as f:
            downloader=MediaIoBaseDownload(f,request,chunksize=8<<20)
            done=False
            while not done:_,done=downloader.next_chunk(num_retries=3)
        if temporary.stat().st_size!=record['size'] or checksum(temporary)!=record['sha256']:
            raise ValueError('Download checksum mismatch: '+name)
        temporary.replace(path)
        print('verified',name,flush=True)
    download('manifest.json');manifest=json.loads((root/'manifest.json').read_text())
    needed=['checksums.json','encoding_spec.json','item_encoding.parquet','user_vocabulary.parquet','candidates.parquet','audit.json','input_feature_audit.json','README.md']
    needed += [x['path'] for x in manifest['train_shards']+manifest['validation_shards']]
    if any(n.startswith('sealed_test/') for n in needed):raise ValueError('Test leakage in download selection')
    for name in needed:download(name)
    extension=release.get('feature_extension')
    if extension:
        for record in extension['files']:
            if record['relative'].startswith('sealed_test/'):continue
            name='features-v2/'+record['relative']
            files[name]={**record,'relative':name}
            download(name)
    return {'dataset':release['dataset_version'],'downloaded_required_files':len(needed)+1,'test_targets_downloaded':False}
