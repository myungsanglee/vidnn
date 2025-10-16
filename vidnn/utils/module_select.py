# from torch import nn, optim
# from vidnn.utils import LOGGER
# from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
# from vidnn.utils.lr_scheduler import CosineAnnealingWarmUpRestarts, YoloLR, VidnnScheduler
from vidnn.data.datamodule import YoloDataModule
from vidnn.models.detect.mobilenetv3_large_100_yolo import MobileNetV3Yolo
from vidnn.module.yolo_detector import YoloDetector


def get_data_module(cfg):
    data_module = None
    task = cfg["task"]
    if task == "detect":
        data_module = YoloDataModule(cfg)

    return data_module


def get_model(cfg):
    model = None
    model_name = cfg["model"]
    task = cfg["task"]
    if model_name == "mobilenetv3_large_100_yolo" and task == "detect":
        model = MobileNetV3Yolo(num_classes=len(cfg["names"]))

    return model


def get_model_module(model, cfg, steps_per_epoch):
    model_module = None
    task = cfg["task"]
    if task == "detect":
        model_module = YoloDetector(model=model, cfg=cfg, steps_per_epoch=steps_per_epoch)

    return model_module


# def get_optimizer(model, name="auto", lr=0.001, momentum=0.9, decay=1e-5):
#     """
#     Construct an optimizer for the given model.

#     Args:
#         model (torch.nn.Module): The model for which to build an optimizer.
#         name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
#             based on the number of iterations.
#         lr (float, optional): The learning rate for the optimizer.
#         momentum (float, optional): The momentum factor for the optimizer.
#         decay (float, optional): The weight decay for the optimizer.

#     Returns:
#         (torch.optim.Optimizer): The constructed optimizer.
#     """
#     g = [], [], []  # optimizer parameter groups
#     bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
#     for module_name, module in model.named_modules():
#         for param_name, param in module.named_parameters(recurse=False):
#             fullname = f"{module_name}.{param_name}" if module_name else param_name
#             if "bias" in fullname:  # bias (no decay)
#                 g[2].append(param)
#             elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
#                 # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
#                 g[1].append(param)
#             else:  # weight (with decay)
#                 g[0].append(param)

#     optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
#     name = {x.lower(): x for x in optimizers}.get(name.lower())
#     if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
#         optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
#     elif name == "RMSProp":
#         optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
#     elif name == "SGD":
#         optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
#     else:
#         raise NotImplementedError(f"Optimizer '{name}' not found in list of available optimizers {optimizers}. ")

#     optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
#     optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
#     LOGGER.info(
#         f"'optimizer:' {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
#         f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
#     )
#     return optimizer


# def get_scheduler(scheduler_name, optim, **kwargs):
#     scheduler_dict = {
#         "multi_step": MultiStepLR,
#         "cosine_annealing_warm_restarts": CosineAnnealingWarmRestarts,
#         "cosine_annealing_warm_up_restarts": CosineAnnealingWarmUpRestarts,
#         "yolo_lr": YoloLR,
#         "ultralytics_lr_v2": VidnnScheduler,
#     }
#     scheduler = scheduler_dict.get(scheduler_name)
#     if scheduler:
#         return scheduler(optim, **kwargs)
