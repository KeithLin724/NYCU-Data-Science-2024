import json
import os
import time
import sys

from torch import no_grad
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from ModelHW import ImageDataset, ModelHW
from tqdm import tqdm


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

    def run(self, file_name: str):
        start_time = time.time()
        assert os.path.exists(file_name)

        with open(file_name, mode="r", encoding="utf-8") as f:
            data_read = json.load(f)

        images_list = data_read["image_paths"]

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
        with open(file=App.OUTPUT_FILE_NAME, mode="w", encoding="utf-8") as f_output:
            json.dump(result_dict, f_output, indent=4)

        end_time = time.time()
        print(f"Output file: {App.OUTPUT_FILE_NAME}")
        print(f"Run time: {end_time - start_time : .3f} s")
        return


if __name__ == "__main__":

    app = App()
    app.run(file_name=sys.argv[1])
