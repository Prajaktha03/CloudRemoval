"""Unit tests for Multimodal Generator, PatchGAN Discriminator, and Baselines."""

import pytest
import torch

from models.discriminator import PatchGANDiscriminator
from models.generator import MultimodalUNet
from models.unet_baseline import OpticalOnlyUNet


def test_generator_forward_and_backward():
    """Verify Generator output shape, range [0, 1], and gradient computation."""
    batch_size = 2
    h, w = 64, 64
    opt_channels = 13
    sar_channels = 2

    model = MultimodalUNet(
        optical_channels=opt_channels,
        sar_channels=sar_channels,
        use_mask=True,
        base_channels=16,
    )

    opt = torch.rand(batch_size, opt_channels, h, w, requires_grad=True)
    sar = torch.rand(batch_size, sar_channels, h, w, requires_grad=True)
    mask = torch.randint(0, 2, (batch_size, 1, h, w)).float()

    output = model(opt, sar, mask)

    assert output.shape == (batch_size, opt_channels, h, w)
    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0
    assert torch.isfinite(output).all()

    # Test backward pass
    loss = output.sum()
    loss.backward()
    assert opt.grad is not None
    assert torch.isfinite(opt.grad).all()


def test_discriminator_forward():
    """Verify PatchGAN Discriminator output patch map shape and finite values."""
    batch_size = 2
    h, w = 64, 64
    opt_channels = 13
    sar_channels = 2

    disc = PatchGANDiscriminator(
        optical_channels=opt_channels,
        sar_channels=sar_channels,
        use_mask=True,
        base_channels=16,
        n_layers=3,
    )

    opt_cond = torch.rand(batch_size, opt_channels, h, w)
    sar_cond = torch.rand(batch_size, sar_channels, h, w)
    candidate = torch.rand(batch_size, opt_channels, h, w)
    mask = torch.zeros(batch_size, 1, h, w)

    logits = disc(opt_cond, sar_cond, candidate, mask)
    # Output should be (B, 1, H_patch, W_patch)
    assert logits.dim() == 4
    assert logits.shape[0] == batch_size
    assert logits.shape[1] == 1
    assert torch.isfinite(logits).all()


def test_optical_only_baseline():
    """Verify OpticalOnlyUNet ablation model runs without SAR."""
    batch_size = 2
    h, w = 64, 64
    opt_channels = 13

    model = OpticalOnlyUNet(
        optical_channels=opt_channels,
        use_mask=True,
        base_channels=16,
    )

    opt = torch.rand(batch_size, opt_channels, h, w)
    mask = torch.zeros(batch_size, 1, h, w)

    output = model(opt, mask)
    assert output.shape == (batch_size, opt_channels, h, w)
    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0
