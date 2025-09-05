import torch
from torch import nn
from torch.optim import SGD
import pytorch_lightning as pl

from vidnn.models.yolo.detect.mobilenetv3_yolov8 import YOLOv8Head
from vidnn.utils.loss import YoloDetectionLoss
from vidnn.utils.module_select import get_optimizer, get_scheduler

# from utils.yolov3_utils import MeanAveragePrecision


class YoloDetector(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.cfg = cfg
        self.model = model
        # self.loss_fn = YoloDetectionLoss()
        # self.loss_fn = YoloV3LossV3(cfg['num_classes'], cfg['anchors'], cfg['input_size'])
        # self.map_metric = MeanAveragePrecision(cfg["num_classes"], cfg["anchors"], cfg["input_size"], cfg["conf_threshold"])

        # Build strides
        m = self.model.head
        if isinstance(m, YOLOv8Head):
            s = 256
            m.inplace = True

            self.model.eval()  # Avoid changing batch statistics until training begins
            m.training = True  # Setting it to True to properly return strides
            m.stride = torch.tensor([s / x.shape[-2] for x in self.model(torch.zeros(1, 3, s, s))])  # forward
            self.model.train()  # Set model back to training(default) mode
            m.bias_init()  # only run once

        # Init weights, biases
        self.initialize_weights()

        self.tloss = None
        self.vloss = None

    def forward(self, x):
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        batch = self.preprocess_batch(batch)
        loss, loss_items = self.forward(batch)

        self.tloss = (self.tloss * batch_idx + loss_items) / (batch_idx + 1) if self.tloss is not None else loss_items
        loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
        self.tloss = self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)

        batch_size = batch["img"].shape[0]
        self.log("train_loss", self.tloss.sum(), prog_bar=True, logger=True, batch_size=batch_size)

        return loss.sum()

    # def on_validation_epoch_start(self):
    #     self.map_metric.reset_states()

    def validation_step(self, batch, batch_idx):
        batch = self.preprocess_batch(batch)
        loss, loss_items = self.forward(batch)

        self.vloss = (self.vloss * batch_idx + loss_items) / (batch_idx + 1) if self.vloss is not None else loss_items
        loss_length = self.vloss.shape[0] if len(self.vloss.shape) else 1
        self.vloss = self.vloss if loss_length > 1 else torch.unsqueeze(self.vloss, 0)

        batch_size = batch["img"].shape[0]
        self.log("val_loss", self.vloss.sum(), prog_bar=True, logger=True, batch_size=batch_size)

        # self.map_metric.update_state(batch["annot"], [p3, p4, p5])

    # def on_validation_epoch_end(self):
    #     map = self.map_metric.result()
    #     self.log("val_mAP", map, prog_bar=True, logger=True)

    def configure_optimizers(self):
        cfg = self.hparams.cfg

        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in self.model.modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)

        optim = SGD(g0, lr=cfg["optimizer_options"]["lr"], momentum=cfg["optimizer_options"]["momentum"], nesterov=True)

        optim.add_param_group({"params": g1, "weight_decay": cfg["optimizer_options"]["weight_decay"]})  # add g1 with weight_decay
        optim.add_param_group({"params": g2})  # add g2 (biases)

        # optim = get_optimizer(cfg["optimizer"], self.model.parameters(), **cfg["optimizer_options"])

        try:
            scheduler = get_scheduler(cfg["scheduler"], optim, **cfg["scheduler_options"])

            return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        except KeyError:
            return optim

    def preprocess_batch(self, batch):
        """
        Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (Dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (Dict): Preprocessed batch with normalized images.
        """
        if getattr(self, "device", None) is None:
            self.device = next(self.parameters()).device
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        # if self.args.multi_scale:
        #     imgs = batch["img"]
        #     sz = (
        #         random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
        #         // self.stride
        #         * self.stride
        #     )  # size
        #     sf = sz / max(imgs.shape[2:])  # scale factor
        #     if sf != 1:
        #         ns = [
        #             math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
        #         ]  # new shape (stretched to gs-multiple)
        #         imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
        #     batch["img"] = imgs
        return batch

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return YoloDetectionLoss(self)

    def initialize_weights(self):
        """Initialize model weights to random values."""
        for m in self.model.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
                m.inplace = True
