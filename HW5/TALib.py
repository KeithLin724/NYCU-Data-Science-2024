import cv2
import pandas as pd
import os
import glob


class TALib:
    def __init__(self):
        return

    @staticmethod
    def dump_to_summit_format(
        folder_name: str = "predictions",
        save_filename: str = "submission.csv",
    ):

        img_list = sorted(list(glob.glob(os.path.join(folder_name, "*.png"))))
        output = []
        for img_name in img_list:
            img = cv2.imread(img_name)
            flatten_img = img.reshape(-1).tolist()
            str_list = [str(i) for i in flatten_img]
            output.append(
                [
                    int(os.path.basename(img_name).replace(".png", "")),
                    " ".join(str_list),
                ]
            )

        df = pd.DataFrame(output, columns=["img_id", "label"])
        df.to_csv(save_filename, index=False)

        return
