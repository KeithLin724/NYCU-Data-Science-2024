import argparse
import glob
import os

import cv2
import pandas as pd

output = []


def parse_args():
    parser = argparse.ArgumentParser(description="Create submission file for gold")
    parser.add_argument("--pred", type=str, default="predictions")
    parser.add_argument("--save-file", type=str, default="submission.csv")
    return parser.parse_args()


def main(args):
    img_list = sorted(list(glob.glob(os.path.join(args.pred, "*.png"))))

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
    df.to_csv(args.save_file, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)