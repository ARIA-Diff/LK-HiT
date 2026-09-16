import numpy as np
import pytest
import torch

from lkhit.losses import AsymmetricLoss, BinaryCrossEntropy, ClassBalancedCrossEntropy, class_balanced_weights, total_loss


def test_asymmetric_loss_is_finite_and_label_mean() -> None:
    loss = AsymmetricLoss(gamma_pos=0.0, gamma_neg=2.0, margin=0.05)
    logits = torch.tensor([[4.0, -4.0, 0.0], [0.0, 3.0, -2.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    value = loss(logits, targets)
    assert value.ndim == 0
    assert torch.isfinite(value)
    assert value.item() > 0


def test_asymmetric_loss_downweights_easy_negatives() -> None:
    loss = AsymmetricLoss(gamma_pos=0.0, gamma_neg=2.0, margin=0.05)
    hard = loss(torch.tensor([[2.0]]), torch.tensor([[0.0]]))
    easy = loss(torch.tensor([[-6.0]]), torch.tensor([[0.0]]))
    assert easy.item() < hard.item()


def test_bce_matches_torch() -> None:
    logits = torch.randn(4, 7)
    targets = torch.randint(0, 2, (4, 7)).float()
    ours = BinaryCrossEntropy()(logits, targets)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(ours, ref)


def test_class_balanced_weights_sum_to_n_classes() -> None:
    counts = np.array([10, 100, 1000, 1])
    weights = class_balanced_weights(counts, beta=0.999)
    assert weights.numel() == 4
    assert torch.allclose(weights.sum(), torch.tensor(4.0), atol=1e-5)
    assert weights[3] > weights[2]


def test_class_balanced_ce_accepts_integer_targets() -> None:
    loss = ClassBalancedCrossEntropy(np.array([50, 200, 10]), beta=0.999)
    logits = torch.randn(5, 3)
    targets = torch.tensor([0, 1, 2, 1, 0])
    value = loss(logits, targets)
    assert torch.isfinite(value)


def test_total_loss_adds_aux() -> None:
    main = torch.tensor(1.5)
    aux = torch.tensor(0.4)
    assert total_loss(main, aux, 1.0).item() == pytest.approx(1.9)
    assert total_loss(main, aux, 0.0).item() == pytest.approx(1.5)
    assert total_loss(main, None, 1.0).item() == pytest.approx(1.5)
