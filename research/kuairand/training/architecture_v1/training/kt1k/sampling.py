"""Training-target frequency correction; never fit on validation/test feedback."""
import hashlib
import numpy as np
import torch


def configure_sampling(model,source,mode,alpha=1.0,smoothing=0.0):
    if mode not in ('none','logq'):
        raise ValueError('sampling_correction must be none or logq')
    if isinstance(alpha,bool) or not isinstance(alpha,(int,float)) or not 0 <= alpha <= 1:
        raise ValueError('sampling_logq_alpha must be a number in [0, 1]')
    if isinstance(smoothing,bool) or not isinstance(smoothing,(int,float)) or smoothing < 0:
        raise ValueError('sampling_logq_smoothing must be a nonnegative number')
    model.sampling_correction=mode
    model.sampling_logq_alpha=float(alpha)
    model.sampling_logq_smoothing=float(smoothing)
    if mode=='none':
        model.sampling_report={'mode':'none','alpha':0.0,'smoothing':0.0}
        return
    values=source.values
    if not getattr(source,'positive_nonrandom_validated',False):
        raise ValueError('Q requires only positive nonrandom train queries')
    ids,counts=np.unique(values['item_id'],return_counts=True)
    raw=model.item.raw_ids.detach().cpu().numpy()
    rows=np.searchsorted(raw,ids)
    if len(ids)==0 or np.any(rows>=len(raw)) or not np.array_equal(raw[rows],ids):
        raise ValueError('Train target is missing from catalog')
    probabilities=np.zeros(len(raw)+1,dtype=np.float32)
    probabilities[rows+1]=(counts+smoothing)/(source.rows+smoothing*len(ids))
    model.register_buffer('sampling_probability',torch.from_numpy(probabilities).to(model.item.raw_ids.device),persistent=False)
    packed=np.column_stack([ids,counts]).astype('<i8',copy=False)
    model.sampling_report={'mode':'logq','alpha':float(alpha),'smoothing':float(smoothing),'source':'positive nonrandom train queries only',
        'train_queries':source.rows,'distinct_targets':len(ids),
        'counts_sha256':hashlib.sha256(packed.tobytes()).hexdigest(),
        'formula':'cosine / temperature - alpha * log((train_target_count + smoothing) / (train_query_count + smoothing * distinct_targets))',
        'evaluation':'uncorrected cosine; unchanged candidates, eligibility and denominator'}


def batch_probability(model,ids):
    if getattr(model,'sampling_correction','none')=='none':return None
    rows=model.item.rows(ids)
    return model.sampling_probability[rows]
