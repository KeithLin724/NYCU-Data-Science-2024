import json
import os
import sys
import time

import torchvision.transforms as transforms
from torch import no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from ModelHW import ImageDataset, ModelHW


class App:
    BATCH_SIZE = 20
    HOT_TYPE = 3
    OUTPUT_FILE_NAME = "image_predictions.json"

    def __init__(self) -> None:
        self.model_hw = ModelHW.load_model("./model/mobileNet_v3_v8_28_BOTH.pth")
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @staticmethod
    def read_file(file_name: str):
        """
        This Python function reads a JSON file and returns the "image_paths" data from it.

        :param file_name: The `file_name` parameter is a string that represents the name or path of the file
        that you want to read
        :type file_name: str
        :return: The function `read_file` is returning the value associated with the key "image_paths" from
        the data read from the file specified by the `file_name` parameter.
        """
        with open(file_name, mode="r", encoding="utf-8") as f:
            data_read = json.load(f)

        return data_read["image_paths"]

    def prediction(self, images_list: list[str]) -> list[int]:
        """
        This Python function takes a list of image paths, processes them using a specified transform, makes
        predictions using a model, and returns a dictionary containing the predictions as a list of
        integers.

        :param images_list: The `images_list` parameter is a list of file paths to images that you want to
        make predictions on
        :type images_list: list[str]
        :return: A dictionary containing the key "image_predictions" with a list of integer predictions
        based on the input images is being returned.
        """
        dataset = ImageDataset(images_list, self.transform)
        data_loader = DataLoader(dataset, App.BATCH_SIZE)

        result = []

        with tqdm(data_loader, unit="batch", desc="Running...") as epoch_data_loader:

            for item in epoch_data_loader:
                with no_grad():
                    pred_term = self.model_hw.prediction(item)

                result.extend(pred_term.tolist())

        # mapping typing
        result = [int(item == App.HOT_TYPE) for item in result]

        result_dict = {"image_predictions": result}
        return result_dict

    def run(self, file_name: str):
        start_time = time.time()
        images_list = App.read_file(file_name)

        result_dict = self.prediction(images_list)

        with open(file=App.OUTPUT_FILE_NAME, mode="w", encoding="utf-8") as f_output:
            json.dump(result_dict, f_output, indent=4)

        end_time = time.time()
        print(f"Output file: {App.OUTPUT_FILE_NAME}")
        print(f"Run time: {end_time - start_time : .3f} s")
        return


if __name__ == "__main__":

    app = App()
    app.run(file_name=sys.argv[1])
