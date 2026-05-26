import torch


def _flatten_scores(scores):
    return scores.float().reshape(-1)


def _zero_like_loss(scores):
    return scores.float().sum() * 0.0


def mix_losses(base_loss,aux_loss,aux_weight):
    aux_weight = float(aux_weight)
    if aux_weight<0.0 or aux_weight>1.0:
        raise ValueError('aux_weight must be in [0, 1]')
    return (1.0-aux_weight) * base_loss + aux_weight * aux_loss


def _as_python_dataset_name(dataset_name):
    if isinstance(dataset_name,(list,tuple)):
        if len(dataset_name)!=1:
            raise ValueError(f'Expected one dataset name per batch, got {dataset_name}')
        dataset_name = dataset_name[0]
    return str(dataset_name)


def resolve_dataset_loss_weight(dataset_name,default_weight,summe_weight=None,tvsum_weight=None):
    dataset_name = _as_python_dataset_name(dataset_name)
    if dataset_name=='SumMe' and summe_weight is not None:
        return float(summe_weight)
    if dataset_name=='TVSum' and tvsum_weight is not None:
        return float(tvsum_weight)
    return float(default_weight)


def pairwise_rank_loss(pred_scores,target_scores,min_target_diff=0.0):
    pred_scores = _flatten_scores(pred_scores)
    target_scores = _flatten_scores(target_scores).to(pred_scores.device)
    target_diff = target_scores.unsqueeze(1) - target_scores.unsqueeze(0)
    pair_mask = target_diff > min_target_diff
    if not torch.any(pair_mask):
        return _zero_like_loss(pred_scores)

    pred_diff = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)
    return torch.nn.functional.softplus(-pred_diff[pair_mask]).mean()


def shot_pairwise_rank_loss(pred_scores,target_scores,shot_index,min_target_diff=0.0):
    pred_scores = _flatten_scores(pred_scores)
    target_scores = _flatten_scores(target_scores).to(pred_scores.device)
    shot_index = shot_index.long().reshape(-1).to(pred_scores.device)
    if pred_scores.numel()!=shot_index.numel():
        raise ValueError('shot_index length must match predicted score length')

    unique_shots = torch.unique(shot_index,sorted=True)
    pred_shots = []
    target_shots = []
    for shot_id in unique_shots:
        mask = shot_index==shot_id
        pred_shots.append(pred_scores[mask].mean())
        target_shots.append(target_scores[mask].mean())

    if len(pred_shots)<2:
        return _zero_like_loss(pred_scores)

    pred_shots = torch.stack(pred_shots)
    target_shots = torch.stack(target_shots)
    return pairwise_rank_loss(pred_shots,target_shots,min_target_diff=min_target_diff)


class MultiAnnotationLoss(torch.nn.Module):
    def __init__(self,precision_ema=0.9,precision_min=0.1,precision_max=10.0,warmup_epochs=0,eps=1e-6):
        super().__init__()
        self.precision_ema = precision_ema
        self.precision_min = precision_min
        self.precision_max = precision_max
        self.warmup_epochs = warmup_epochs
        self.eps = eps
        self.precision = {}
        self.base_loss = torch.nn.MSELoss()

    def _precision_key(self,dataset_name,annot_scores):
        return f'{dataset_name}:{annot_scores.shape[0]}'

    def _current_precision(self,key,annot_scores):
        if key not in self.precision:
            return torch.ones(annot_scores.shape[0],device=annot_scores.device,dtype=annot_scores.dtype)
        return self.precision[key].to(device=annot_scores.device,dtype=annot_scores.dtype)

    def _update_precision(self,key,pred_scores,annot_scores):
        with torch.no_grad():
            residual = (pred_scores.detach().unsqueeze(0) - annot_scores).pow(2).mean(dim=1)
            new_precision = residual.add(self.eps).reciprocal()
            new_precision = torch.clamp(new_precision,self.precision_min,self.precision_max).cpu()
            if key in self.precision:
                old_precision = self.precision[key]
                new_precision = self.precision_ema * old_precision + (1.0-self.precision_ema) * new_precision
            self.precision[key] = new_precision

    def forward(self,pred_scores,avg_target,annot_scores,dataset_name,epoch=0):
        pred_scores = _flatten_scores(pred_scores)
        avg_target = _flatten_scores(avg_target).to(pred_scores.device)
        annot_scores = annot_scores.float().to(pred_scores.device)
        if annot_scores.dim()==1:
            annot_scores = annot_scores.unsqueeze(0)
        if annot_scores.shape[-1]!=pred_scores.numel():
            raise ValueError('annot_scores length must match predicted score length')

        if epoch<self.warmup_epochs:
            return self.base_loss(pred_scores,avg_target)

        key = self._precision_key(dataset_name,annot_scores)
        precision = self._current_precision(key,annot_scores)
        residual = (pred_scores.unsqueeze(0) - annot_scores).pow(2)
        loss = (residual * precision.unsqueeze(1)).sum()
        loss = loss / (precision.sum() * pred_scores.numel()).clamp_min(self.eps)
        self._update_precision(key,pred_scores,annot_scores)
        return loss

    def summary(self):
        return {key: value.tolist() for key,value in self.precision.items()}
