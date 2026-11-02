"""Exhaustive FP32 single-target ranking; no sampled-negative metric."""
import time

import numpy as np
import torch


def ranked_metrics(scores,targets,raw_ids,eligible_after,query_times,k=100,block=65536):
    """Rank by score descending / raw item ID ascending; uncovered targets miss.

    Target score is gathered from the SAME GEMM result, avoiding differences
    between a separate dot-product implementation and retrieval GEMM rounding.
    """
    target_scores=scores.gather(1,targets[:,None]).squeeze(1)
    target_ids=raw_ids[targets]
    covered=eligible_after[targets]<=query_times
    rank=torch.ones(len(targets),device=scores.device,dtype=torch.long)
    for start in range(0,len(raw_ids),block):
        end=min(start+block,len(raw_ids))
        values=scores[:,start:end]
        allowed=eligible_after[start:end][None,:]<=query_times[:,None]
        ahead=(values>target_scores[:,None])|((values==target_scores[:,None])&(raw_ids[start:end][None,:]<target_ids[:,None]))
        rank+=(ahead&allowed).sum(1)
    hit=covered&(rank<=k)
    gain=torch.where(hit,1/torch.log2(rank.double()+1),0)
    return rank,hit,gain,covered


@torch.no_grad()
def evaluate(model,queries,eligible,settings,device,rank_path=None):
    model.eval()
    begin=time.perf_counter()
    ids=model.item.raw_ids
    vectors=torch.empty((len(ids),settings['output_dim']),device=device,dtype=torch.float32)
    for start in range(0,len(ids),settings['item_batch_size']):
        vectors[start:start+settings['item_batch_size']]=model.item(ids[start:start+settings['item_batch_size']])
        if start==0 or start%(settings['item_batch_size']*20)==0:
            print(f'catalog encoding {min(start+settings["item_batch_size"],len(ids))}/{len(ids)}',flush=True)
    hits,gains,coverage,count=0,0.0,0,0
    rank_rows=[]
    for _,cpu in queries.batches(settings['query_batch_size']):
        batch={n:t.to(device) for n,t in cpu.items()}
        u=model.users(batch)
        # Exhaustive: e.g. 128 x 4.37M FP32 scores occupy ~2.24 GB on GPU.
        scores=u.float()@vectors.T
        positions=torch.searchsorted(ids,batch['item_id'])
        if torch.any(positions>=len(ids)) or not torch.equal(ids[positions],batch['item_id']):
            raise ValueError('Validation target is absent from catalog')
        ranks,h,g,c=ranked_metrics(scores,positions,ids,eligible,batch['time_ms'],settings['k'])
        hits+=int(h.sum());gains+=float(g.sum());coverage+=int(c.sum());count+=len(u)
        rank_rows.append(np.column_stack([cpu['event_id'].numpy(),ranks.cpu().numpy(),c.cpu().numpy()]))
        del scores
        if count% (settings['query_batch_size']*100)==0:
            print(f'evaluation {count}/{queries.rows}',flush=True)
    if count!=queries.rows:
        raise ValueError('Incomplete validation traversal')
    if rank_path is not None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows=np.concatenate(rank_rows)
        pq.write_table(pa.table({'event_id':rows[:,0],'target_rank':rows[:,1],'covered':rows[:,2].astype(bool)}),rank_path,compression='zstd')
    return {'recall_at_100':hits/count,'ndcg_at_100':gains/count,'target_coverage':coverage/count,'queries':count,'hits':hits,'evaluation_seconds':time.perf_counter()-begin,'catalog_size':len(ids),'ranking':'exhaustive FP32; raw ID ascending tie break; repeats allowed; uncovered targets count as misses'}
