import os
import sys

import numpy as np
import torch

sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))

from experiment_losses import MultiAnnotationLoss, mix_losses, pairwise_rank_loss, resolve_dataset_loss_weight, shot_pairwise_rank_loss
from dataset import _normalize_tvsum_annotation_scores


def _assert_finite_grad(tensor):
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()


def test_tvsum_annotation_normalization_matches_h5_scale():
    raw_scores = np.array([1.0,2.0,3.0,4.0,5.0],dtype=np.float32)
    normalized = _normalize_tvsum_annotation_scores(raw_scores)
    expected = np.array([0.0,0.25,0.5,0.75,1.0],dtype=np.float32)
    assert np.allclose(normalized,expected)


def test_mix_losses_interpolates_and_validates_weight():
    base_loss = torch.tensor(2.0)
    aux_loss = torch.tensor(6.0)
    mixed = mix_losses(base_loss,aux_loss,0.25)
    assert torch.isclose(mixed,torch.tensor(3.0))
    try:
        mix_losses(base_loss,aux_loss,1.5)
    except ValueError:
        pass
    else:
        raise AssertionError('mix_losses should reject weights outside [0, 1]')


def test_pairwise_rank_loss_backward():
    pred = torch.tensor([0.1,0.2,0.3],requires_grad=True)
    target = torch.tensor([3.0,2.0,1.0])
    loss = pairwise_rank_loss(pred,target)
    assert torch.isfinite(loss)
    assert loss.item()>0
    loss.backward()
    _assert_finite_grad(pred)


def test_shot_pairwise_rank_loss_backward():
    pred = torch.tensor([0.1,0.2,0.7,0.8],requires_grad=True)
    target = torch.tensor([1.0,1.0,3.0,3.0])
    shot_index = torch.tensor([0,0,1,1])
    loss = shot_pairwise_rank_loss(pred,target,shot_index)
    assert torch.isfinite(loss)
    loss.backward()
    _assert_finite_grad(pred)


def test_multi_annotation_loss_backward_and_summary():
    loss_fn = MultiAnnotationLoss(precision_ema=0.0,precision_min=0.1,precision_max=10.0)
    pred = torch.tensor([0.2,0.4,0.6],requires_grad=True)
    avg_target = torch.tensor([0.3,0.5,0.7])
    annot_scores = torch.tensor([
        [0.2,0.5,0.7],
        [0.4,0.4,0.8]
    ])
    loss = loss_fn(pred,avg_target,annot_scores,'TVSum',epoch=1)
    assert torch.isfinite(loss)
    loss.backward()
    _assert_finite_grad(pred)
    assert 'TVSum:2' in loss_fn.summary()


def test_resolve_dataset_loss_weight_uses_optional_overrides():
    assert resolve_dataset_loss_weight(['SumMe'],0.25,summe_weight=0.5,tvsum_weight=0.0)==0.5
    assert resolve_dataset_loss_weight(['TVSum'],0.25,summe_weight=0.5,tvsum_weight=0.0)==0.0
    assert resolve_dataset_loss_weight(['SumMe'],0.25,summe_weight=None,tvsum_weight=0.0)==0.25
    assert resolve_dataset_loss_weight(['TVSum'],0.25,summe_weight=0.5,tvsum_weight=None)==0.25


if __name__=='__main__':
    test_tvsum_annotation_normalization_matches_h5_scale()
    test_mix_losses_interpolates_and_validates_weight()
    test_pairwise_rank_loss_backward()
    test_shot_pairwise_rank_loss_backward()
    test_multi_annotation_loss_backward_and_summary()
    test_resolve_dataset_loss_weight_uses_optional_overrides()
