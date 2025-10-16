from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from vidnn.utils import LOGGER


def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


def model_info(model, detailed=False, verbose=True, imgsz=640):
    """
    Print and return detailed model information layer by layer.

    Args:
        model (nn.Module): Model to analyze.
        detailed (bool, optional): Whether to print detailed layer information.
        verbose (bool, optional): Whether to print model information.
        imgsz (int | list, optional): Input image size.

    Returns:
        n_l (int): Number of layers.
        n_p (int): Number of parameters.
        n_g (int): Number of gradients.
        flops (float): GFLOPs.
    """
    if not verbose:
        return
    n_p = get_num_params(model)  # number of parameters
    n_g = get_num_gradients(model)  # number of gradients
    layers = __import__("collections").OrderedDict((n, m) for n, m in model.named_modules() if len(m._modules) == 0)
    n_l = len(layers)  # number of layers
    if detailed:
        h = f"{'layer':>5}{'name':>40}{'type':>20}{'gradient':>10}{'parameters':>12}{'shape':>20}{'mu':>10}{'sigma':>10}"
        LOGGER.info(h)
        for i, (mn, m) in enumerate(layers.items()):
            mn = mn.replace("module_list.", "")
            mt = m.__class__.__name__
            if len(m._parameters):
                for pn, p in m.named_parameters():
                    LOGGER.info(
                        f"{i:>5g}{f'{mn}.{pn}':>40}{mt:>20}{p.requires_grad!r:>10}{p.numel():>12g}{str(list(p.shape)):>20}{p.mean():>10.3g}{p.std():>10.3g}{str(p.dtype).replace('torch.', ''):>15}"
                    )
            else:  # layers with no learnable params
                LOGGER.info(f"{i:>5g}{mn:>40}{mt:>20}{False!r:>10}{0:>12g}{str([]):>20}{'-':>10}{'-':>10}{'-':>15}")

    flops = get_flops(model, imgsz)  # imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320]
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    LOGGER.info(f"{model_name} summary{fused}: {n_l:,} layers, {n_p:,} parameters, {n_g:,} gradients{fs}")
    return n_l, n_p, n_g, flops


def get_num_params(model):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """Return the total number of parameters with gradients in a YOLO model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def get_flops(model, imgsz=640):
    """
    Calculate FLOPs (floating point operations) for a model in billions.

    Attempts two calculation methods: first with a stride-based tensor for efficiency,
    then falls back to full image size if needed (e.g., for RTDETR models). Returns 0.0
    if thop library is unavailable or calculation fails.

    Args:
        model (nn.Module): The model to calculate FLOPs for.
        imgsz (int | list, optional): Input image size.

    Returns:
        (float): The model FLOPs in billions.
    """
    try:
        import thop
    except ImportError:
        thop = None  # conda support without 'ultralytics-thop' installed

    if not thop:
        return 0.0  # if not installed return 0.0 GFLOPs

    try:
        model = de_parallel(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]  # expand if int/float
        try:
            # Method 1: Use stride-based input tensor
            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  # max stride
            im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # stride GFLOPs
            return flops * imgsz[0] / stride * imgsz[1] / stride  # imgsz GFLOPs
        except Exception:
            # Method 2: Use actual image size (required for RTDETR models)
            im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # imgsz GFLOPs
    except Exception:
        return 0.0


def is_parallel(model):
    """
    Return True if model is of type DP or DDP.

    Args:
        model (nn.Module): Model to check.

    Returns:
        (bool): True if model is DataParallel or DistributedDataParallel.
    """
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):
    """
    De-parallelize a model: return single-GPU model if model is of type DP or DDP.

    Args:
        model (nn.Module): Model to de-parallelize.

    Returns:
        (nn.Module): De-parallelized model.
    """
    return model.module if is_parallel(model) else model
