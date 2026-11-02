"""Schema-v2 data reader; explicitly selects all eleven item features."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def required_files(manifest):
    names=['manifest.json','encoding_spec.json','item_encoding.parquet','user_vocabulary.parquet','candidates.parquet']
    return names+[e['path'] for e in manifest['train_shards']+manifest['validation_shards']]


def verify(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    checks=json.loads((root/'checksums.json').read_text())
    if manifest['schema_version']!=2 or manifest['dataset']!='KuaiRand-1K':
        raise ValueError('Require the prepared KuaiRand-1K schema-v2 release')
    for name in required_files(manifest):
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=checks[name]:
            raise ValueError('Dataset checksum mismatch: '+name)
    return manifest


def extension(root,manifest):
    directory=Path(root)/'features-v2'
    spec=json.loads((directory/'feature_spec.json').read_text())
    checks=json.loads((directory/'checksums.json').read_text())
    if spec['base_manifest_sha256']!=sha(Path(root)/'manifest.json'):
        raise ValueError('Feature extension belongs to different base data')
    required=['feature_spec.json','item_extra.parquet','tag_vocabulary.parquet']
    required += [e['path'] for e in manifest['train_shards']+manifest['validation_shards']]
    for name in required:
        if sha(directory/name)!=checks[name]:raise ValueError('Feature checksum mismatch: '+name)
    return spec


class Queries:
    """Compact host arrays (~0.4 GB train+validation), deterministic row shuffle."""
    def __init__(self,root,entries,length=50,feature_spec=None):
        import pyarrow.parquet as pq
        self.length=length
        count=sum(e['rows'] for e in entries)
        self.values={n:np.empty(count,dtype=np.int64) for n in ['event_id','user_id','item_id','time_ms','tab']}
        self.histories=np.full((count,length),-1,dtype=np.int32)
        self.feature_spec=feature_spec
        if feature_spec:
            self.history_feedback=np.zeros((count,length,5),dtype=np.float32)
            self.user_numeric=np.zeros((count,6),dtype=np.float32)
        cursor=0
        for entry in entries:
            f=pq.ParquetFile(Path(root)/entry['path'])
            if f.metadata.num_rows!=entry['rows']:
                raise ValueError('Shard row count mismatch')
            columns=list(self.values)+['history_item_ids','history_max_available_ms','long_view','is_random']
            for batch in f.iter_batches(batch_size=32768,columns=columns):
                end=cursor+len(batch)
                for name in self.values:
                    self.values[name][cursor:end]=batch.column(name).to_numpy()
                if not np.all(batch.column('long_view').to_numpy()==1) or not np.all(batch.column('is_random').to_numpy()==0):
                    raise ValueError('Only nonrandom positive queries may reach training/validation')
                now=self.values['time_ms'][cursor:end]
                if np.any(batch.column('history_max_available_ms').to_numpy()>=now):
                    raise ValueError('Noncausal history')
                lists=batch.column('history_item_ids')
                offsets=lists.offsets.to_numpy().astype(np.int64)
                lens=np.diff(offsets)
                if lens.max(initial=0)>length:
                    raise ValueError('Oversized history')
                flat=lists.values.to_numpy()[offsets[0]:offsets[-1]]
                if flat.size and (flat.min()<0 or flat.max()>np.iinfo(np.int32).max):
                    raise ValueError('Invalid raw history IDs')
                rows=np.repeat(np.arange(len(batch)),lens)
                cols=np.arange(len(flat))-np.repeat(offsets[:-1]-offsets[0],lens)+length-np.repeat(lens,lens)
                self.histories[cursor+rows,cols]=flat
                cursor=end
        if feature_spec:
            cursor=0
            fields=['history_time_ms','history_play_ms','history_completion','history_completion_known','history_like']
            for entry in entries:
                f=pq.ParquetFile(Path(root)/'features-v2'/entry['path'])
                if f.metadata.num_rows!=entry['rows']:raise ValueError('Feature row count mismatch')
                for batch in f.iter_batches(batch_size=32768):
                    end=cursor+len(batch)
                    if not np.array_equal(batch.column('event_id').to_numpy(),self.values['event_id'][cursor:end]):
                        raise ValueError('Feature/query identity mismatch')
                    lengths=(self.histories[cursor:end]>=0).sum(1)
                    now=self.values['time_ms'][cursor:end]
                    for i,name in enumerate(fields):
                        lists=batch.column(name);offsets=lists.offsets.to_numpy().astype(np.int64);lens=np.diff(offsets)
                        if not np.array_equal(lens,lengths):raise ValueError('History feedback alignment mismatch')
                        flat=lists.values.to_numpy()[offsets[0]:offsets[-1]].astype(np.float64)
                        rows=np.repeat(np.arange(len(batch)),lens)
                        cols=np.arange(len(flat))-np.repeat(offsets[:-1]-offsets[0],lens)+length-np.repeat(lens,lens)
                        if i==0:
                            if np.any(flat>=now[rows]):raise ValueError('Future history time')
                            flat=np.log1p((now[rows]-flat)/3600000)
                        elif i==1:flat=np.log1p(flat/1000)
                        self.history_feedback[cursor+rows,cols,i]=flat
                    numeric=np.asarray(batch.column('user_numeric').to_pylist(),dtype=np.float32)
                    self.user_numeric[cursor:end]=(numeric-np.asarray(feature_spec['user_numeric_mean']))/np.asarray(feature_spec['user_numeric_std'])
                    cursor=end
            if not np.isfinite(self.history_feedback).all() or not np.isfinite(self.user_numeric).all():
                raise ValueError('Nonfinite behavior features')
        assert cursor==count
        self.rows=count
        self.positive_nonrandom_validated=True

    def batches(self,batch_size,seed=None,start=0):
        order=np.arange(self.rows) if seed is None else np.random.default_rng(seed).permutation(self.rows)
        for ordinal,begin in enumerate(range(start*batch_size,self.rows,batch_size),start):
            selected=order[begin:begin+batch_size]
            result={k:torch.from_numpy(v[selected]) for k,v in self.values.items()}
            result['history_item_ids']=torch.from_numpy(self.histories[selected]).long()
            result['history_mask']=result['history_item_ids']>=0
            if self.feature_spec:
                result['history_feedback']=torch.from_numpy(self.history_feedback[selected])
                result['user_numeric']=torch.from_numpy(self.user_numeric[selected])
            yield ordinal+1,result


def load_item_data(root,spec):
    import pyarrow.parquet as pq
    fields=['item_id']+['cat_'+n for n in spec['categorical_names']]+['num_'+n for n in spec['numeric_names']]
    t=pq.read_table(Path(root)/'item_encoding.parquet',columns=fields)
    raw=t['item_id'].to_numpy().copy()
    if len(raw)<1 or np.any(np.diff(raw)<=0):
        raise ValueError('Item lookup must be unique and sorted')
    cat=np.column_stack([t['cat_'+n].to_numpy() for n in spec['categorical_names']])
    num=np.column_stack([t['num_'+n].to_numpy() for n in spec['numeric_names']]).astype(np.float32)
    for i,card in enumerate(spec['cardinalities']):
        if cat[:,i].min()<0 or cat[:,i].max()>=card:
            raise ValueError('Invalid categorical index')
    if not np.isfinite(num).all():
        raise ValueError('Nonfinite item numeric feature')
    return raw,cat,num
