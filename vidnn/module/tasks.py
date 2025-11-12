import math
import random
import torch
from torch import nn, optim
import pytorch_lightning as pl
import numpy as np

from vidnn.nn.head import DetectHeadV1, OBB
from vidnn.utils.loss import DetectionLoss, OBBLoss
from vidnn.utils.lr_scheduler import VidnnScheduler
from vidnn.utils import LOGGER, ops
from vidnn.utils.metrics import DetMetrics, box_iou, OBBMetrics, batch_probiou
from vidnn.data.utils import load_image_from_source, letterbox


class DetectorModule(pl.LightningModule):
    def __init__(self, model, cfg, steps_per_epoch):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model
        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch

        # Build strides
        m = self.model.head
        if isinstance(m, DetectHeadV1):
            s = 256
            m.inplace = True

            self.model.eval()  # Avoid changing batch statistics until training begins
            m.training = True  # Setting it to True to properly return strides
            preds = model(torch.zeros(1, 3, s, s))
            preds = preds[0] if isinstance(m, (OBB)) else preds
            m.stride = torch.tensor([s / x.shape[-2] for x in preds])  # forward
            self.model.train()  # Set model back to training(default) mode
            m.bias_init()  # only run once

        # Init weights, biases
        self.initialize_weights()

        self.tloss = None
        self.vloss = None

        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics(names=cfg["names"])
        self.nc = len(cfg["names"])
        self.names = cfg["names"]

        self.criterion = self.init_criterion()

    def forward(self, x):
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x)
        return self.model(x)

    def predict(self, source, conf=0.25, iou=0.6, imgsz=640, max_det=300):
        # preprocess
        img, orig_img = load_image_from_source(source, imgsz)
        img = letterbox(img, imgsz)
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[::-1] if img.shape[0] == 3 else img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        if getattr(self, "device", None) is None:
            self.device = next(self.parameters()).device
        img = img.to(self.device, non_blocking=True).float() / 255

        # inference
        preds = self.forward(img)

        # Postprocess
        preds = ops.non_max_suppression(
            preds,
            conf_thres=conf,
            iou_thres=iou,
            max_det=max_det,
        )
        pred = preds[0]
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return pred, orig_img

    def on_train_epoch_start(self):
        self.tloss = None
        cfg = self.hparams.cfg
        total_epochs = cfg["epochs"]
        close_mosaic = cfg["close_mosaic"]
        current_epoch = self.current_epoch
        if current_epoch == (total_epochs - close_mosaic):
            self.trainer.train_dataloader.dataset.close_mosaic()
            LOGGER.info("Closing dataloader mosaic")

    def training_step(self, batch, batch_idx):
        # Preprocess
        batch = self.preprocess_batch(batch)

        # Forward
        loss, loss_items = self.forward(batch)

        # Get Loss
        self.tloss = (self.tloss * batch_idx + loss_items) / (batch_idx + 1) if self.tloss is not None else loss_items
        loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
        self.tloss = self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)

        # Log
        batch_size = batch["img"].shape[0]
        self.log(
            "train_loss",
            self.tloss.sum(),
            prog_bar=True,
            # logger=True,
            batch_size=batch_size,
        )

        return loss.sum()

    def on_validation_epoch_start(self):
        self.vloss = None
        self.seen = 0

    def validation_step(self, batch, batch_idx):
        # Preprocess
        batch = self.preprocess_batch(batch)

        # Inference
        preds = self.forward(batch["img"])

        # Loss
        loss_items = self.loss(batch, preds)[1]

        # Postprocess
        preds = self.postprocess(preds)

        self.vloss = (self.vloss * batch_idx + loss_items) / (batch_idx + 1) if self.vloss is not None else loss_items
        loss_length = self.vloss.shape[0] if len(self.vloss.shape) else 1
        self.vloss = self.vloss if loss_length > 1 else torch.unsqueeze(self.vloss, 0)

        batch_size = batch["img"].shape[0]
        self.log(
            "val_loss",
            self.vloss.sum(),
            prog_bar=True,
            # logger=True,
            batch_size=batch_size,
        )

        self.update_metrics(preds, batch)

    def on_validation_epoch_end(self):
        stats = self.get_stats()
        stats = {k: round(float(v), 5) for k, v in stats.items()}
        precision = stats["metrics/precision(B)"]
        recall = stats["metrics/recall(B)"]
        mAP50 = stats["metrics/mAP50(B)"]
        mAP50_95 = stats["metrics/mAP50-95(B)"]
        fitness = stats["fitness"]
        self.print_results()

        # self.log("metrics/mAP50(B)", mAP50, prog_bar=False, logger=True)
        # self.log("metrics/mAP50-95(B)", mAP50_95, prog_bar=False, logger=True)
        # self.log("metrics/precision(B)", precision, prog_bar=False, logger=True)
        # self.log("metrics/recall(B)", recall, prog_bar=False, logger=True)
        # self.log("fitness", fitness, prog_bar=False, logger=False)

        self.log("metrics/mAP50(B)", mAP50)
        self.log("metrics/mAP50-95(B)", mAP50_95)
        self.log("metrics/precision(B)", precision)
        self.log("metrics/recall(B)", recall)
        self.log("fitness", fitness)

    def configure_optimizers(self):
        cfg = self.hparams.cfg

        # Optimizer
        accumulate = max(round(cfg["nbs"] / cfg["batch_size"]), 1)  # accumulate loss before optimizing
        weight_decay = cfg["weight_decay"] * cfg["batch_size"] * accumulate / cfg["nbs"]  # scale weight_decay
        optim = self.build_optimizer(
            model=self.model,
            name=cfg["optimizer"],
            lr=cfg["lr0"],
            momentum=cfg["momentum"],
            decay=weight_decay,
        )

        # Scheduler
        warmup_steps = max(round(cfg["warmup_epochs"] * self.steps_per_epoch), 100) if cfg["warmup_epochs"] > 0 else -1  # warmup iterations
        scheduler = VidnnScheduler(
            optimizer=optim,
            total_epochs=cfg["epochs"],
            steps_per_epoch=self.steps_per_epoch,
            warmup_steps=warmup_steps,
            lrf=cfg["lrf"],
            cos_lr=cfg["cos_lr"],
        )

        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

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

        if not self.training:  # Validation
            for k in {"batch_idx", "cls", "bboxes"}:
                batch[k] = batch[k].to(self.device)

        elif self.training and self.cfg["multi_scale"]:
            imgs = batch["img"]
            sz = random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride)) // self.stride * self.stride  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs

        return batch

    def postprocess(self, preds):
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (List[Dict[str, torch.Tensor]]): Processed predictions after NMS, where each dict contains
                'bboxes', 'conf', 'cls', and 'extra' tensors.
        """
        outputs = ops.non_max_suppression(
            preds,
            self.cfg["conf"] if self.cfg["conf"] is not None else 0.001,
            self.cfg["iou"],
            nc=0,
            multi_label=True,
            agnostic=False,
            max_det=self.cfg["max_det"],
            rotated=False,
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        # if getattr(self, "criterion", None) is None:
        #     self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return DetectionLoss(self)

    def initialize_weights(self):
        """Initialize model weights to random values."""
        # for m in self.model.modules():
        modules_to_init = nn.ModuleList([self.model.neck, self.model.head])
        for m in modules_to_init.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
                m.inplace = True

    def update_metrics(self, preds, batch):
        """
        Update metrics with new predictions and ground truth.

        Args:
            preds (List[Dict[str, torch.Tensor]]): List of predictions from the model.
            batch (Dict[str, Any]): Batch data containing ground truth.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = len(predn["cls"]) == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )
            # Evaluate
            if no_pred:
                continue

    def _prepare_batch(self, si, batch):
        """
        Prepare a batch of images and annotations for validation.

        Args:
            si (int): Batch index.
            batch (Dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (Dict[str, Any]): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
        }

    def _prepare_pred(self, pred):
        """
        Prepare predictions for evaluation against ground truth.

        Args:
            pred (Dict[str, torch.Tensor]): Post-processed predictions from the model.

        Returns:
            (Dict[str, torch.Tensor]): Prepared predictions in native space.
        """
        if self.cfg["single_cls"]:
            pred["cls"] *= 0
        return pred

    def _process_batch(self, preds, batch):
        """
        Return correct prediction matrix.

        Args:
            preds (Dict[str, torch.Tensor]): Dictionary containing prediction data with 'bboxes' and 'cls' keys.
            batch (Dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' and 'cls' keys.

        Returns:
            (Dict[str, np.ndarray]): Dictionary containing 'tp' key with correct prediction matrix of shape (N, 10) for 10 IoU levels.
        """
        if len(batch["cls"]) == 0 or len(preds["cls"]) == 0:
            return {"tp": np.zeros((len(preds["cls"]), self.niou), dtype=bool)}
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def match_predictions(self, pred_classes, true_classes, iou):
        """
        Match predictions to ground truth objects using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape (N,).
            true_classes (torch.Tensor): Target class indices of shape (M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground truth.

        Returns:
            (torch.Tensor): Correct tensor of shape (N, 10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def get_stats(self):
        """
        Calculate and return metrics statistics.

        Returns:
            (Dict[str, Any]): Dictionary containing metrics results.
        """
        self.metrics.process()
        self.metrics.clear_stats()
        return self.metrics.results_dict

    def print_results(self):
        """Print training/validation set metrics per class."""
        LOGGER.info(("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)"))
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(
            pf
            % (
                "all",
                self.seen,
                self.metrics.nt_per_class.sum(),
                *self.metrics.mean_results(),
            )
        )
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in detect set, can not compute metrics without labels")

        # Print results per class
        # if self.nc > 1 and len(self.metrics.stats) and not self.training:
        #     for i, c in enumerate(self.metrics.ap_class_index):
        #         LOGGER.info(
        #             pf
        #             % (
        #                 self.names[c],
        #                 self.metrics.nt_per_image[c],
        #                 self.metrics.nt_per_class[c],
        #                 *self.metrics.class_result(i),
        #             )
        #         )

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5):
        """
        Construct an optimizer for the given model.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations.
            lr (float, optional): The learning rate for the optimizer.
            momentum (float, optional): The momentum factor for the optimizer.
            decay (float, optional): The weight decay for the optimizer.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizers = {
            "Adam",
            "Adamax",
            "AdamW",
            "NAdam",
            "RAdam",
            "RMSProp",
            "SGD",
            "auto",
        }
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f"Optimizer '{name}' not found in list of available optimizers {optimizers}. ")

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"'optimizer:' {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer


class OBBModule(DetectorModule):
    def __init__(self, model, cfg, steps_per_epoch):
        super().__init__(model, cfg, steps_per_epoch)
        self.metrics = OBBMetrics(names=cfg["names"])

    def predict(self, source, conf=0.25, iou=0.6, imgsz=640, max_det=300):
        # preprocess
        img, orig_img = load_image_from_source(source, imgsz)
        img = letterbox(img, imgsz)
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[::-1] if img.shape[0] == 3 else img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        if getattr(self, "device", None) is None:
            self.device = next(self.parameters()).device
        img = img.to(self.device, non_blocking=True).float() / 255

        # inference
        preds = self.forward(img)

        # Postprocess
        preds = ops.non_max_suppression(
            preds,
            conf_thres=conf,
            iou_thres=iou,
            max_det=max_det,
            nc=len(self.cfg["names"]),
            rotated=True,
        )
        pred = preds[0]
        rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
        obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
        return obb, orig_img

    def init_criterion(self):
        """Initialize the loss criterion for the OBBModel."""
        return OBBLoss(self)

    def postprocess(self, preds):
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (List[Dict[str, torch.Tensor]]): Processed predictions after NMS, where each dict contains
                'bboxes', 'conf', 'cls', and 'extra' tensors.
        """
        outputs = ops.non_max_suppression(
            preds,
            self.cfg["conf"] if self.cfg["conf"] is not None else 0.001,
            self.cfg["iou"],
            nc=self.nc,
            multi_label=True,
            agnostic=False,
            max_det=self.cfg["max_det"],
            rotated=True,
        )
        preds = [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]
        for pred in preds:
            pred["bboxes"] = torch.cat([pred["bboxes"], pred.pop("extra")], dim=-1)  # concatenate angle
        return preds

    def _prepare_batch(self, si, batch):
        """
        Prepare batch data for OBB validation with proper scaling and formatting.

        Args:
            si (int): Batch index to process.
            batch (dict[str, Any]): Dictionary containing batch data with keys:
                - batch_idx: Tensor of batch indices
                - cls: Tensor of class labels
                - bboxes: Tensor of bounding boxes
                - ori_shape: Original image shapes
                - img: Batch of images
                - ratio_pad: Ratio and padding information

        Returns:
            (dict[str, Any]): Prepared batch data with scaled bounding boxes and metadata.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # target boxes
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
        }

    def _process_batch(self, preds, batch):
        """
        Compute the correct prediction matrix for a batch of detections and ground truth bounding boxes.

        Args:
            preds (dict[str, torch.Tensor]): Prediction dictionary containing 'cls' and 'bboxes' keys with detected
                class labels and bounding boxes.
            batch (dict[str, torch.Tensor]): Batch dictionary containing 'cls' and 'bboxes' keys with ground truth
                class labels and bounding boxes.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing 'tp' key with the correct prediction matrix as a numpy
                array with shape (N, 10), which includes 10 IoU levels for each detection, indicating the accuracy
                of predictions compared to the ground truth.
        """
        if len(batch["cls"]) == 0 or len(preds["cls"]) == 0:
            return {"tp": np.zeros((len(preds["cls"]), self.niou), dtype=bool)}
        iou = batch_probiou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}
