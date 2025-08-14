import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os

from .dataset import YoloDataset


class YoloDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.train_path = os.path.join(self.cfg["path"], self.cfg["train"])
        self.val_path = os.path.join(self.cfg["path"], self.cfg["val"])
        self.test_path = (
            os.path.join(self.cfg["path"], self.cfg["test"])
            if self.cfg["test"] is not None
            else None
        )
        self.batch_size = self.cfg["batch_size"]
        self.num_workers = self.cfg["workers"]

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = YoloDataset(
                self.train_path,
                self.cfg["imgsz"],
                augment=True,
                hyp=self.cfg,
            )
            self.val_dataset = YoloDataset(
                self.val_path,
                self.cfg["imgsz"],
                augment=False,
                hyp=self.cfg,
            )

        if stage == "test" or stage is None:
            if self.test_path is not None:
                self.test_dataset = YoloDataset(
                    self.test_path,
                    self.cfg["imgsz"],
                    augment=False,
                    hyp=self.cfg,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=True,
        )
