import torch
import torch.nn as nn

import lightning as L


class DataModuleGAN(L.LightningDataModule):
    def __init__(self):
        super().__init__()
