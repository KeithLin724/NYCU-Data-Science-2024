import torch
import torch.nn as nn

import lightning as L

# ref:
## https://lightning.ai/docs/pytorch/stable/data/datamodule.html


class DataModuleGAN(L.LightningDataModule):
    def __init__(self):
        super().__init__()
