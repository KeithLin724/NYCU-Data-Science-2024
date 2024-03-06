import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.models import mobilenet_v3_large


class Model:
    def __init__(self, model_path: str) -> None:
        self._device = torch.device("cpu")
