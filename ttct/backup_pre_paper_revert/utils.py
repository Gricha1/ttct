import random
import numpy as np
import pickle
from num2words import num2words
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch 

class KLLoss(torch.nn.Module):
    def __init__(self, error_metric=torch.nn.KLDivLoss(reduction='batchmean')):
        super().__init__()
        self.error_metric = error_metric

    def forward(self, prediction, label):
        prediction = torch.nan_to_num(prediction, nan=0.0, posinf=1e4, neginf=-1e4)
        label = torch.nan_to_num(label, nan=0.0, posinf=1.0, neginf=0.0)
        probs1 = F.log_softmax(prediction, dim=1)
        probs2 = F.softmax(label * 10.0, dim=1).clamp(min=1e-8, max=1.0)
        probs2 = probs2 / probs2.sum(dim=1, keepdim=True).clamp(min=1e-8)
        loss = self.error_metric(probs1, probs2)
        return loss


class MultiPositiveContrastiveLoss(torch.nn.Module):
    """Soft-target CE on in-batch logits (no logit clamp). mask[i,j]=1 => positive pair."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = torch.nan_to_num(prediction, nan=0.0, posinf=1e4, neginf=-1e4)
        logits = logits / max(self.temperature, 1e-6)
        log_probs = F.log_softmax(logits, dim=1)
        target = mask.float()
        target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return -(target * log_probs).sum(dim=1).mean()


def _tl_key(item):
    """
    Hashable key for template-language items in gen_mask.
    Must not use tuple(str) — that splits a string into characters and breaks matching
    (Craftax energy: TLs = [natural language string]).
    """
    if isinstance(item, tuple):
        return item
    return (item,)


def align_obs_act(obs, act):
    """Match obs_t with act_t (drop final-only obs when len(obs)==len(act)+1)."""
    obs = np.asarray(obs, dtype=np.float32)
    act = np.asarray(act, dtype=np.float32)
    if len(obs) == len(act) + 1:
        obs = obs[:-1]
    n = max(min(len(obs), len(act)), 1)
    return obs[:n], act[:n], n


def gen_mask(batch_TLs):
    batch_sets = [{_tl_key(x) for x in sublist} for sublist in batch_TLs]

    unique_TLs = [random.choice(sublist) for sublist in batch_TLs]
    num_unique = len(unique_TLs)
    matrix = np.zeros((len(batch_TLs), num_unique), dtype=np.float32)

    TL_to_index = {}
    for j, TL in enumerate(unique_TLs):
        key = _tl_key(TL)
        if key in TL_to_index:
            TL_to_index[key].append(j)
        else:
            TL_to_index[key] = [j]
    for i, TL_set in enumerate(batch_sets):
        for key in TL_set:
            if key in TL_to_index:
                for j in TL_to_index[key]:
                    matrix[i, j] = 1.0

    count = np.sum(matrix)
    return unique_TLs, matrix, count


def gen_mask_from_nl(batch_nls):
    """One column per unique NL; row i is one-hot. Use with forward(nl_texts=...) -> [B, U]."""
    unique_nls = list(dict.fromkeys(batch_nls))
    nl_to_j = {nl: j for j, nl in enumerate(unique_nls)}
    matrix = np.zeros((len(batch_nls), len(unique_nls)), dtype=np.float32)
    for i, nl in enumerate(batch_nls):
        matrix[i, nl_to_j[nl]] = 1.0
    return unique_nls, matrix, float(matrix.sum())


class U3TDataset(Dataset):
    def __init__(self, data):
        self.data=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        obs = self.data[index][0]
        act = self.data[index][1]
        TLs = self.data[index][2] #template language
        length = self.data[index][3]
        NLs = self.data[index][4] #trajectory level natural language
        return obs,act,TLs,length,NLs
    
def split_dataset(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    x_train,x_test = train_test_split(data , train_size=0.8)
    return U3TDataset(x_train),U3TDataset(x_test)

    


