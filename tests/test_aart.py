import torch

from aart import AART8_BRANCHES, AARTConv2d, AARTUNet2D, sample_rt_offsets, soft_feature_center


def test_soft_center_and_sampling_shapes():
    x = torch.rand(2, 4, 16, 16, requires_grad=True)
    center = soft_feature_center(x)
    assert center.shape == (2, 2)
    samples = sample_rt_offsets(x, [(-1, 0), (0, 1), (1, 0)])
    assert samples.shape == (2, 4, 3, 16, 16)
    samples.mean().backward()
    assert x.grad is not None


def test_aart_conv_uses_final_eight_branches():
    conv = AARTConv2d(8, 12)
    assert conv.branch_names == AART8_BRANCHES
    x = torch.rand(2, 8, 20, 20, requires_grad=True)
    y = conv(x)
    assert y.shape == (2, 12, 20, 20)
    y.mean().backward()
    assert x.grad is not None
    assert conv.last_center is not None


def test_aart_unet_forward_backward():
    model = AARTUNet2D(in_channels=1, num_classes=4, base_channels=8)
    x = torch.rand(1, 1, 64, 64)
    y = model(x)
    assert y.shape == (1, 4, 64, 64)
    y.mean().backward()
