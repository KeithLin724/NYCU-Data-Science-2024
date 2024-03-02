import asyncio
import json
import os
import time

import cv2
import httpx
import numpy as np
import pandas as pd
from rich import print

from CrawlerHW import CrawlerHW


class CrawlerData:
    TRAN_DATA_FILE = "./tran_data.jsonl"
    DATA_FOLDER = "./data"

    def __init__(self) -> None:
        with open(file=CrawlerData.TRAN_DATA_FILE, mode="r", encoding="utf-8") as f:
            data = f.readlines()
        data = [json.loads(item) for item in data]

        self.table = pd.DataFrame(data)

        # make folder
        if not os.path.exists(CrawlerData.DATA_FOLDER):
            os.makedirs(CrawlerData.DATA_FOLDER)

    @staticmethod
    def save_path(file_name: str) -> str:
        return os.path.join(CrawlerData.DATA_FOLDER, file_name)

    @staticmethod
    def process_hot_number(hot_numbers: list):
        "process hot number"

        def tran_to_number(hot_number):

            if isinstance(hot_number, str):
                if hot_number[0] == "X":
                    second_item = item if (item := hot_number[1:]) != "X" else "10"
                    return int("-" + second_item)
                if "爆" in hot_number:
                    return 100

            return int(hot_number)

        return [tran_to_number(number) for number in hot_numbers]

    @staticmethod
    def bytes_to_image(image_bytes: bytes) -> cv2.typing.MatLike:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image)  # , cv2.COLOR_BGR2RGB
        return image

    async def fetch_image(self, url: str, client: httpx.AsyncClient):
        "fetch image"
        temp = CrawlerData.save_path(url.split("/")[-1])
        if os.path.exists(temp):
            return {
                "image_link": url,
                "state_code": 200,
                "reason_phase": "OK",
                "image_state": True,  # make sure image is exists
                "file_path": temp,
            }

        await asyncio.sleep(CrawlerHW.get_random_wait_time())
        try:
            image_content = await client.get(
                url, headers=CrawlerHW.get_random_user_agent(), timeout=60
            )
        except Exception as e:
            return {
                "image_link": url,
                "state_code": 0,
                "reason_phase": str(e),
                "image_state": False,  # make sure image is exists
                "file_path": "",
            }

        file_path, image = "", None
        # get image
        if image_content.status_code == 200:
            image_file_name = url.split("/")[-1]
            file_path = CrawlerData.save_path(image_file_name)
            image = CrawlerData.bytes_to_image(image_content.content)
            if image is not None:
                try:
                    cv2.imwrite(file_path, image)
                except Exception as e:
                    return {
                        "image_link": url,
                        "state_code": image_content.status_code,
                        "reason_phase": str(e),
                        "image_state": False,  # make sure image is exists
                        "file_path": "",
                    }
            else:
                file_path = ""

        print(
            f"Process image: {url}, state_code: {image_content.status_code}, reason_phase: {image_content.reason_phrase}             ",
            end="\r",
        )

        return {
            "image_link": url,
            "state_code": image_content.status_code,
            "reason_phase": image_content.reason_phrase,
            "image_state": image is not None,  # make sure image is exists
            "file_path": file_path,
        }

    def process_table_in_hot_number(self):
        self.table["hotNumber"] = CrawlerData.process_hot_number(
            self.table["hotNumber"]
        )

        # label
        self.table["is_hot"] = (self.table["hotNumber"] > 35).astype(int)
        print(self.table)
        self.table.to_csv("tran_data.csv", index=False)
        print("Finish: save file in tran_data.csv")
        return

    async def fetch_all_image(self):
        image_link_data_lists = self.table["image_link"]

        # test
        # image_link_data_lists = image_link_data_lists[:10]

        async with httpx.AsyncClient() as client:
            task_list = [
                self.fetch_image(task_url, client) for task_url in image_link_data_lists
            ]

            result_images = await CrawlerHW.gather(*task_list)

        self.image_table = pd.DataFrame(result_images)
        self.image_table.to_csv("tran_data_image.csv", index=False)
        print("\nFinish: save file in tran_data_image.csv")
        return

    def to_big_table(self):
        self.big_table = pd.merge(
            self.table, self.image_table, on="image_link", how="inner"
        )

        self.big_table.to_csv("tran_data_all.csv", index=False)
        print("Finish: save file in tran_data_all.csv")

    async def run(self):
        start_time = time.time()
        # process hot number
        self.process_table_in_hot_number()
        # get the image
        await self.fetch_all_image()
        # to big table
        self.to_big_table()
        end_time = time.time()

        print(f"Run Time: {end_time-start_time}")
        return


if __name__ == "__main__":
    asyncio.run(CrawlerData().run())
