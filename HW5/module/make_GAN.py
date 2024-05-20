import torch
import torch.nn as nn

import lightning as L


class Gan(L.LightningModule):
    def __init__(self):
        super().__init__()
