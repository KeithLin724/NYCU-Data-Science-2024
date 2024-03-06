from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import mobilenet_v3_large


class ModelHW:
    def __init__(self, model) -> None:
        self._device = torch.device("cpu")

        self._model = model
        self._model.eval()

        self._classes = ["-10_1", "19_35", "2_6", "36_100", "7_18"]

    @property
    def model(self):
        return self._model

    @classmethod
    def load_model(cls, file_path: str):
        model = mobilenet_v3_large()

        num_features = model.classifier[-1].in_features
        # output only two class
        model.classifier[-1] = nn.Linear(num_features, 5)

        model.load_state_dict(
            torch.load(
                file_path,
                map_location="cpu",
            )
        )

        return cls(model)

    def prediction(self, obj_tensor: torch.Tensor):
        pred_prob = self._model(obj_tensor)
        pred = torch.max(pred_prob, 1).indices
        # pred = pred.item()
        return pred


class ImageDataset(Dataset):
    def __init__(self, image_paths: list[str], transform: transforms.Compose) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> Any:
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image
