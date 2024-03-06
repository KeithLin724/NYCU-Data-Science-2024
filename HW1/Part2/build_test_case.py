import os
import json

from main import App
from sklearn.metrics import f1_score
import random as rd
import pandas as pd


def sample(folder_name: str, k: int, type_: str) -> pd.DataFrame:
    all_image_list = os.listdir(folder_name)
    # make a full path
    all_image_path = [os.path.join(folder_name, item) for item in all_image_list]

    # sample
    random_sample = rd.sample(all_image_path, k=k)

    result = {"Image_path": random_sample, "Type": [type_] * len(random_sample)}

    return pd.DataFrame(result)


def main() -> None:
    hot_image_folder, boo_image_folder = "./data/HOT", "./data/BOO"

    hot_sample_table, boo_sample_table = (
        sample(hot_image_folder, k=50, type_=1),
        sample(boo_image_folder, k=50, type_=0),
    )

    testing_table = pd.concat([hot_sample_table, boo_sample_table])
    shuffled_table = testing_table.sample(frac=1).reset_index(drop=True)

    # testing
    app = App()
    pred_result = app.prediction(shuffled_table["Image_path"].tolist())
    pred_result = pred_result["image_predictions"]

    true_list = shuffled_table["Type"].tolist()

    f1 = f1_score(true_list, pred_result)
    acc = (shuffled_table["Type"] == pd.Series(pred_result)).mean() * 100
    print(f"f1 scope {f1:.3f}, acc {acc:.3f}")

    return


if __name__ == "__main__":
    main()
