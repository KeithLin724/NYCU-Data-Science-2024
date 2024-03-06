from bs4 import BeautifulSoup
from typing import Callable
from functools import reduce
from rich import print
import pandas as pd
import time


class PttCrawlerTool:
    @staticmethod
    def page_to_simple_dict(html_str: str, func_list: list[Callable]):
        soup = BeautifulSoup(html_str, "html.parser")
        page_dict = {}

        page_dict = reduce(lambda a, b: a | b, [func(soup) for func in func_list])

        return page_dict

    @staticmethod
    def page_content(soup: BeautifulSoup):
        return soup.find("div", class_="bbs-screen bbs-content", id="main-content")

    @staticmethod
    def page_header_data(soup: BeautifulSoup) -> dict[str, str]:
        body_data = PttCrawlerTool.page_content(soup)

        # get header data
        header_data = body_data.find_all("div", class_="article-metaline")

        tab_list = [
            str(line.find("span", class_="article-meta-tag").string)
            for line in header_data
        ]

        value_list = [
            str(line.find("span", class_="article-meta-value").string)
            for line in header_data
        ]

        header_dict = dict(zip(tab_list, value_list))

        if "時間" not in header_dict:
            # get text body #https://www.ptt.cc/bbs/Beauty/M.1690589266.A.166.html
            process_text = body_data.contents[2]
            process_text = process_text.split("\n")[:2]
            process_text = [item.split(":") for item in process_text]
            process_result_dict = {item[0]: item[1] for item in process_text}
            header_dict |= process_result_dict

        pd_time = pd.to_datetime(header_dict["時間"], format="%a %b %d %H:%M:%S %Y")
        header_dict |= {"pd_time": pd_time}

        return header_dict

    @staticmethod
    def page_all_image_link(soup: BeautifulSoup) -> dict:
        body_data = PttCrawlerTool.page_content(soup)
        images_catch_link = body_data.find_all("div", class_="richcontent")

        images_catch_link = [
            item.get("src")
            for image in images_catch_link
            if (item := image.find("img"))
        ]

        # image link lists
        image_link = body_data.find_all("a")
        image_link = [
            link_str
            for link in image_link
            if any(
                substring in (link_str := str(link.string))
                for substring in [".png", ".jpg", "jpeg", ".gif"]
            )
        ]

        return {
            "image_catch_list": images_catch_link,
            "image_link": image_link,
        }


import httpx

PTT_OVER_18_HEADER = {"cookie": "over18=1"}

result = httpx.get(
    "https://www.ptt.cc/bbs/Beauty/M.1672554775.A.108.html", headers=PTT_OVER_18_HEADER
)
s = time.time()
page_dict = PttCrawlerTool.page_to_simple_dict(
    result.text,
    func_list=[
        PttCrawlerTool.page_header_data,
        # PttCrawlerTool.page_all_image_link,
    ],
)
e = time.time()
print(page_dict)
print(f"time :{e-s}")
