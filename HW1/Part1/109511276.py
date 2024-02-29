import asyncio
import json
import random as rd
import sys
import time
from functools import reduce
from typing import Callable

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from rich import print
from tqdm.asyncio import tqdm

SAMPLE_URL = "https://www.ptt.cc/bbs/Beauty/M.1672503968.A.5B5.html"
START_URL = "https://www.ptt.cc/bbs/Beauty/index3662.html"
# START_URL = "https://www.ptt.cc/bbs/Beauty/index3827.html"


PTT_PREV_LINK = "https://www.ptt.cc"
SEARCH_YEAR = "2023"

# make like a human in CrawlerHW

# https://blog.csdn.net/a_123_4/article/details/119718509
USER_AGENT_LIST = [
    # Opera
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36 OPR/26.0.1656.60",
    "Opera/8.0 (Windows NT 5.1; U; en)",
    "Mozilla/5.0 (Windows NT 5.1; U; en; rv:1.8.1) Gecko/20061208 Firefox/2.0.0 Opera 9.50",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; en) Opera 9.50",
    # Firefox
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0",
    "Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:65.0) Gecko/20100101 Firefox/65.0",
    # Safari
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.57.2 (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2",
    # chrome
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 Safari/534.16",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.75 Safari/537.36",
    # 360
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",
    # 淘宝浏览器
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.11 TaoBrowser/2.0 Safari/536.11",
    # 猎豹浏览器
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; LBBROWSER)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E; LBBROWSER)",
    # QQ浏览器
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; QQBrowser/7.0.3698.400)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)",
    # sogou浏览器
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.84 Safari/535.11 SE 2.X MetaSr 1.0",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SV1; QQDownload 732; .NET4.0C; .NET4.0E; SE 2.X MetaSr 1.0)",
    # maxthon浏览器
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Maxthon/4.4.3.4000 Chrome/30.0.1599.101 Safari/537.36",
    # UC浏览器
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.122 UBrowser/4.0.3214.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.100 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 13_3_1 like Mac OS X; zh-CN) AppleWebKit/537.51.1 (KHTML, like Gecko) Mobile/17D50 UCBrowser/12.8.2.1268 Mobile AliApp(TUnionSDK/0.1.20.3)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 8.1.0; OPPO R11t Build/OPM1.171019.011; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/76.0.3809.89 Mobile Safari/537.36 T7/11.19 SP-engine/2.15.0 baiduboxapp/11.19.5.10 (Baidu; P1 8.1.0)",
    "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 13_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 SP-engine/2.14.0 main%2F1.0 baiduboxapp/11.18.0.16 (Baidu; P2 13.3.1) NABar/0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 12_4_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 MicroMessenger/7.0.10(0x17000a21) NetType/4G Language/zh_CN",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.108 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.106 Safari/537.36",
]


USER_AGENT_HEAD = "User-Agent"
PTT_OVER_18_HEADER = {"cookie": "over18=1"}


class CustomEncoder(json.JSONEncoder):
    def encode(self, o):
        if isinstance(o, dict) and len(o) <= 2:
            return (
                "{" + ", ".join(f'"{k}": {json.dumps(v)}' for k, v in o.items()) + "}"
            )
        else:
            return super().encode(o)


class CrawlerHW:
    ARTICLES_FILE_NAME = "articles.jsonl"
    POPULAR_ARTICLES_FILE_NAME = "popular_articles.jsonl"

    ERROR_INPUT_MESSAGE = "please input command -> crawl, push <start_date> <end_date>, popular <start_date> <end_date>, keyword <start_date> <end_date> <keyword>"

    FIRST_TEN = 10
    CHUNK_SIZE = 100

    def __init__(self) -> None:
        pass

    # static function
    @staticmethod
    def page_to_simple_dict(html_str: str, func_list: list[Callable] = None) -> dict:
        """
        format like
        '作者': 'ReiKuromiya (ReiKuromiya)',
        '標題': '[正妹] 周子瑜',
        '時間': 'Sun Jan  1 00:26:06 2023',
        'Year': '2023',
        'Month': 'Jan',
        'Day': '',
        'Week': 'Sun',
        'Time': '00:26:06',
        # 'Body': [(..., ...),  ...]}
        """

        def to_detail_date(date_str: str) -> dict:
            detail_date = date_str.split(" ")

            if "" in detail_date:
                detail_date.remove("")
            return {
                "Year": detail_date[-1],
                "Month": detail_date[1],
                "Day": detail_date[2],
                "Week": detail_date[0],
                "Time": detail_date[-2],
            }

        soup = BeautifulSoup(html_str, "html.parser")

        # get main data
        body_data = soup.find("div", class_="bbs-screen bbs-content", id="main-content")

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

        page_data = header_dict | to_detail_date(header_dict["時間"])

        if func_list is not None:
            addition_dict_list = reduce(
                lambda a, b: a | b, [func(soup) for func in func_list]
            )
            page_data |= addition_dict_list

        return page_data

    @staticmethod
    def recommend_page_to_simple_dict(html_str: str) -> dict:
        """for recommend page like index.html"""

        def item_change_to_dict(div_obj: BeautifulSoup) -> dict:
            hot_number = div_obj.find("span")

            hot_number = str(hot_number.string) if hot_number is not None else 0

            if isinstance(hot_number, str) and hot_number.isdigit():
                hot_number = int(hot_number)

            title = div_obj.find("a")

            title_str, title_url = (
                title.string if title else "",
                title.get("href") if title else "",
            )

            author = div_obj.find("div", class_="author")
            author = str(author.string) if author is not None else ""

            date = div_obj.find("div", class_="date")
            date = str(date.string) if date is not None else ""

            return {
                "Title": title_str,
                "Author": author,
                "HotNumber": hot_number,
                "Date": date,
                "URL": title_url,
            }

        def filter_function(item_dict: dict) -> bool:
            if "[公告]" in item_dict["Title"]:
                return False

            if "Fw:[公告]" in item_dict["Title"]:
                return False

            if item_dict["Title"] == "":
                return False

            if item_dict["URL"] == "":
                return False

            return True

        # header = {"cookie": "over18=1"}
        # result = httpx.get(url=url, headers=header)
        soup = BeautifulSoup(html_str, "html.parser")

        # action button -> button dict
        action_bar_button = soup.find_all("div", class_="btn-group btn-group-paging")[0]
        action_bar_button = action_bar_button.find_all("a")

        location = ["Old", "Prev", "Next", "New"]
        button_link = [item.get("href", "") for item in action_bar_button]

        button_dict = dict(zip(location, button_link))
        ##############################################

        # get recommend list -> detail dict
        recommend_list = soup.find(
            "div", class_="r-list-container action-bar-margin bbs-screen"
        )
        recommend = recommend_list.find_all("div", class_="r-ent")
        recommend_result = [item_change_to_dict(div_obj=item) for item in recommend]
        recommend_after_filter_result = list(filter(filter_function, recommend_result))

        return button_dict | {"Body": recommend_after_filter_result}

    @staticmethod
    def get_random_user_agent() -> dict[str, str]:
        return {USER_AGENT_HEAD: rd.choice(USER_AGENT_LIST)}

    @staticmethod
    def get_header():
        return PTT_OVER_18_HEADER | CrawlerHW.get_random_user_agent()

    @staticmethod
    def get_random_wait_time() -> float:
        return rd.uniform(0.1, 1.1)

    @staticmethod
    def to_full_link(url: str) -> str:
        return PTT_PREV_LINK + url

    @staticmethod
    def to_table_time(raw_str: str) -> str:
        element = [
            item if len((item := part_str.strip())) > 1 else f"0{item}"
            for part_str in raw_str.split("/")
        ]
        return "".join(element)

    @staticmethod
    def dict_save_to_file(save_dict: dict, file_name: str):
        with open(file_name, mode="w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=4, ensure_ascii=False)
        return

    @staticmethod
    def get_sub_table(
        date_start: pd.DatetimeIndex, date_end: pd.DatetimeIndex, table: pd.DataFrame
    ) -> pd.DataFrame:
        return table[(table["date"] >= date_start) & (table["date"] <= date_end)]

    @staticmethod
    def build_task_from_table(
        func: Callable,
        table: pd.DataFrame,
        column_name: str,
        client: httpx.AsyncClient,
        **kw,
    ):
        return [func(item[column_name], client, **kw) for _, item in table.iterrows()]

    @staticmethod
    async def gather(*task):
        "make like asyncio"

        ## cut to chunk
        chunk_task = [
            task[i : i + CrawlerHW.CHUNK_SIZE]
            for i in range(0, len(task), CrawlerHW.CHUNK_SIZE)
        ]

        tasks_result = [await asyncio.gather(*chunk_pack) for chunk_pack in chunk_task]
        # add to result
        tasks_result = sum(tasks_result, [])
        return tasks_result

    ###############################################################

    # main function

    async def craw_page_by_dict(self, small_page_dict: dict, client: httpx.AsyncClient):
        "save file(2) and return is in 2023"
        # make like a human
        await asyncio.sleep(CrawlerHW.get_random_wait_time())
        # print(small_page_dict)
        page_response = await client.get(
            CrawlerHW.to_full_link(small_page_dict["URL"]),
            headers=CrawlerHW.get_header(),
        )

        page_dict = CrawlerHW.page_to_simple_dict(html_str=page_response.text)

        page_dict |= {
            "table_time": CrawlerHW.to_table_time(small_page_dict["Date"]),
            "HotNumber": small_page_dict["HotNumber"],
            "URL": CrawlerHW.to_full_link(small_page_dict["URL"]),
        }
        # print(page_dict)

        return (page_dict["Year"], page_dict["Year"] == SEARCH_YEAR, page_dict)

    async def crawl(self, client: httpx.AsyncClient):

        "get the 2023 year ptt Beauty"

        "page by page to get the data"
        now_page_url = START_URL
        in_range = True

        with open(
            file=CrawlerHW.ARTICLES_FILE_NAME, mode="w", encoding="utf-8"
        ) as f_articles, open(
            file=CrawlerHW.POPULAR_ARTICLES_FILE_NAME, mode="w", encoding="utf-8"
        ) as f_popular:
            pass
        articles_cnt, popular_cnt = 0, 0
        # make like a human -> page by page
        while now_page_url != "":
            if not in_range:
                break

            recommend_page = await client.get(
                now_page_url, headers=CrawlerHW.get_header()
            )

            recommend_simple_dict = CrawlerHW.recommend_page_to_simple_dict(
                html_str=recommend_page.text
            )

            small_page: list[dict] = recommend_simple_dict["Body"]

            small_page_tasks = [
                self.craw_page_by_dict(item, client) for item in small_page
            ]

            # run small page
            small_page_tasks_result = await asyncio.gather(*small_page_tasks)

            with open(
                file=CrawlerHW.ARTICLES_FILE_NAME, mode="a", encoding="utf-8"
            ) as f_articles, open(
                file=CrawlerHW.POPULAR_ARTICLES_FILE_NAME, mode="a", encoding="utf-8"
            ) as f_popular:

                for year, is_in_2023, page_dict in small_page_tasks_result:

                    if year == "2024":
                        in_range = False
                        continue

                    if not is_in_2023:
                        continue

                    file_dict = {
                        "date": page_dict["table_time"],
                        "title": page_dict["標題"],
                        "url": page_dict["URL"],
                    }

                    json.dump(file_dict, f_articles, ensure_ascii=False)
                    f_articles.write("\n")
                    articles_cnt += 1

                    if page_dict["HotNumber"] == "爆":
                        json.dump(file_dict, f_popular, ensure_ascii=False)
                        f_popular.write("\n")
                        popular_cnt += 1

                    now_year = year
                    print(
                        f"Crawling ... Date: {now_year}/{page_dict['Month']}/{page_dict['Day']} Page: {now_page_url}",
                        end="\r",
                    )

            now_page_url = CrawlerHW.to_full_link(recommend_simple_dict["Next"])
            print(
                f"Crawling ... Date: {now_year}/{page_dict['Month']}/{page_dict['Day']} Page: {now_page_url}",
                end="\r",
            )

            # break
        print(
            f"Crawl Finish Date: {now_year}/{page_dict['Month']}/{page_dict['Day']} Page: {now_page_url}"
        )
        print(f"Articles: {articles_cnt}, Popular Articles: {popular_cnt}")

        return

    @staticmethod
    def group_data_by_user_name(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        return (
            df.groupby("user_id")
            .agg(
                {
                    "like_boo_type": "first",
                    "body": lambda x: list(x),
                    "count": "sum",
                }
            )
            .reset_index()
        )

    @staticmethod
    def get_like_boo_count_dict(soup: BeautifulSoup) -> dict:
        """_summary_
        for push function (addition for page_to_simple_dict function)
        Args:
            soup (BeautifulSoup): html obj

        Returns:
            dict: "page_comment_table" , pd.DataFrame
        """

        def push_list_to_dict(item: BeautifulSoup) -> dict:

            span_item = [thing.string for thing in item.find_all("span")]
            span_item = [str(thing).strip() for thing in span_item]

            split_res = span_item[3].split(" ")
            ip, date_str, time_str = "", "", ""
            if len(split_res) == 3:
                ip, date_str, time_str = split_res

            return {
                "like_boo_type": span_item[0],
                "user_id": span_item[1],
                "body": [
                    {
                        "comment": span_item[2].replace(":", "").strip(),
                        "ip": ip,
                        "date": date_str,
                        "time": time_str,
                    }
                ],
                "count": 1,
            }

        push_list = soup.find_all("div", class_="push")
        # print(push_list)

        all_comment = [push_list_to_dict(item=item) for item in push_list]

        like_list = [item for item in all_comment if item["like_boo_type"] == "推"]
        boo_list = [item for item in all_comment if item["like_boo_type"] == "噓"]
        mid_list = [item for item in all_comment if item["like_boo_type"] == "→"]

        like_table = pd.DataFrame(like_list)
        boo_table = pd.DataFrame(boo_list)
        mid_table = pd.DataFrame(mid_list)

        like_table = CrawlerHW.group_data_by_user_name(like_table)
        boo_table = CrawlerHW.group_data_by_user_name(boo_table)
        mid_table = CrawlerHW.group_data_by_user_name(mid_table)

        return {
            "like_table": like_table,
            "boo_table": boo_table,
            "mid_table": mid_table,
        }

    async def craw_page(self, url: str, client: httpx.AsyncClient):
        # make like a human
        await asyncio.sleep(CrawlerHW.get_random_wait_time())

        page_response = await client.get(url, headers=CrawlerHW.get_header())

        page_dict = CrawlerHW.page_to_simple_dict(
            page_response.text, func_list=[CrawlerHW.get_like_boo_count_dict]
        )
        print(f"Process url:{url}", end="\r")

        return {
            "like_table": page_dict["like_table"],
            "boo_table": page_dict["boo_table"],
            "mid_table": page_dict["mid_table"],
        }

    @staticmethod
    def to_pd_date(*date_str):
        date = [f"{SEARCH_YEAR}-{item}" for item in date_str]
        date = pd.to_datetime(date, format="%Y-%m%d").strftime("%Y-%m-%d")
        return date

    @staticmethod
    def load_data_frame_from_file(file_name: str) -> pd.DataFrame:
        with open(file_name, mode="r", encoding="utf-8") as f:
            data_list = f.readlines()

        # line of json
        data_list = [json.loads(item) for item in data_list]
        data_df = pd.DataFrame(data_list)

        # make date is pd.date_time
        data_df["date"] = [f"{SEARCH_YEAR}-{item}" for item in data_df["date"]]
        data_df["date"] = pd.to_datetime(data_df["date"], format="%Y-%m%d")
        return data_df

    async def push(self, client: httpx.AsyncClient, date_start: str, date_end: str):
        # make input to datetime
        file_name = f"push_{date_start}_{date_end}.json"

        date_start, date_end = CrawlerHW.to_pd_date(date_start, date_end)

        print(f"Search range: {date_start} to {date_end}")

        # load jsonl to pd.DataFrame
        print("Loading file...")

        data_df = CrawlerHW.load_data_frame_from_file(CrawlerHW.ARTICLES_FILE_NAME)

        # get range of table
        sub_table = CrawlerHW.get_sub_table(date_start, date_end, data_df)

        # make task
        page_tasks = CrawlerHW.build_task_from_table(
            func=self.craw_page,
            table=sub_table,
            column_name="url",
            client=client,
        )

        print("Crawling...")

        # tasks_result = await asyncio.gather(*page_tasks)  # , unit="page"
        tasks_result = await CrawlerHW.gather(*page_tasks)

        print("\nProcessing...")

        def to_big_table(key: str, table: pd.DataFrame) -> pd.DataFrame:
            "collect all table and take out empty table and return the table is sorted"
            big_table = [item[key] for item in table if not item[key].empty]
            big_table = pd.concat(big_table, ignore_index=True)
            big_table = CrawlerHW.group_data_by_user_name(big_table)
            big_table = big_table.sort_values(
                by=["count", "user_id"], ascending=[False, False]
            )

            return big_table

        def table_to_dict(type_: str, table: pd.DataFrame) -> dict:
            table = table.drop(columns=["like_boo_type", "body"])
            dict_list = table.to_dict("records")
            # first 10
            total, top_10 = len(table), dict_list[: CrawlerHW.FIRST_TEN]

            return {type_: {"total": total, "top10": top_10}}

        like_total_table = to_big_table("like_table", tasks_result)
        boo_total_table = to_big_table("boo_table", tasks_result)
        mid_total_table = to_big_table("mid_table", tasks_result)

        like_total_dict = table_to_dict("push", like_total_table)
        boo_total_dict = table_to_dict("boo", boo_total_table)
        mid_total_dict = table_to_dict("mid", mid_total_table)

        display_dict = like_total_dict | boo_total_dict

        CrawlerHW.dict_save_to_file(display_dict, file_name)

        print(f"File save in: {file_name}")

        return

    @staticmethod
    def get_images_from_page(soup: BeautifulSoup) -> dict:
        "for get page image line (a addition function for page to simple dict function)"
        body_data = soup.find("div", class_="bbs-screen bbs-content", id="main-content")
        # image src
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

    async def craw_popular_page(self, url: str, client: httpx.AsyncClient):
        await asyncio.sleep(CrawlerHW.get_random_wait_time())
        popular_page_response = await client.get(url, headers=CrawlerHW.get_header())

        page_dict = CrawlerHW.page_to_simple_dict(
            popular_page_response.text,
            func_list=[CrawlerHW.get_images_from_page],
        )
        print(f"Process url:{url}", end="\r")
        return {
            "image_catch_list": page_dict["image_catch_list"],
            "image_link": page_dict["image_link"],
        }

    async def popular(self, client: httpx.AsyncClient, date_start: str, date_end: str):
        file_name = f"popular_{date_start}_{date_end}.json"
        date_start, date_end = CrawlerHW.to_pd_date(date_start, date_end)

        print(f"Search range: {date_start} to {date_end}")
        print("Loading file ...")

        data_df = CrawlerHW.load_data_frame_from_file(
            CrawlerHW.POPULAR_ARTICLES_FILE_NAME
        )

        sub_table = CrawlerHW.get_sub_table(date_start, date_end, data_df)

        popular_page_tasks = CrawlerHW.build_task_from_table(
            func=self.craw_popular_page,
            table=sub_table,
            column_name="url",
            client=client,
        )

        print("Crawling...")

        popular_page_tasks_result = await CrawlerHW.gather(*popular_page_tasks)

        print("\nProcessing...")

        image_urls = [item["image_link"] for item in popular_page_tasks_result]
        image_urls = sum(image_urls, [])

        result_dict = {
            "number_of_popular_articles": len(sub_table),
            "image_urls": image_urls,
        }

        CrawlerHW.dict_save_to_file(result_dict, file_name)

        print(f"File save in {file_name}")

        return

    @staticmethod
    def get_body_content(soup: BeautifulSoup) -> dict:
        body_data = soup.find("div", class_="bbs-screen bbs-content", id="main-content")

        body_text = body_data.text.strip()

        index_end = body_text.find("※ 發信站")
        # must have ※ 發信站
        body_text = body_text[:index_end]

        return {"is_can_use": bool(index_end != -1), "body_content": body_text}

    async def craw_page_by_keyword(self, url: str, client: httpx.AsyncClient, **kw):
        await asyncio.sleep(CrawlerHW.get_random_wait_time())

        page_response = await client.get(url, headers=CrawlerHW.get_header())

        page_dict = CrawlerHW.page_to_simple_dict(
            page_response.text,
            func_list=[
                CrawlerHW.get_body_content,
                CrawlerHW.get_images_from_page,
            ],
        )

        if not page_dict["is_can_use"]:
            return None
        print(f"Process url:{url}", end="\r")
        return (
            page_dict["image_link"]
            if kw["kw"]["keyword"] in page_dict["body_content"]
            else None
        )

    async def keyword(
        self,
        client: httpx.AsyncClient,
        date_start: str,
        date_end: str,
        keyword_str: str,
    ):
        keyword_str = keyword_str.strip()
        file_name = f"keyword_{date_start}_{date_end}_{keyword_str}.json"

        date_start, date_end = CrawlerHW.to_pd_date(date_start, date_end)
        print(f"Search range: {date_start} to {date_end}")

        print("Loading file ...")
        data_df = CrawlerHW.load_data_frame_from_file(CrawlerHW.ARTICLES_FILE_NAME)

        sub_table = CrawlerHW.get_sub_table(date_start, date_end, data_df)

        keyword_page_dict = CrawlerHW.build_task_from_table(
            func=self.craw_page_by_keyword,
            table=sub_table,
            column_name="url",
            client=client,
            kw={"keyword": keyword_str},
        )

        tasks_result = await CrawlerHW.gather(*keyword_page_dict)

        print("\nProcessing...")
        all_image_urls = sum([item for item in tasks_result if item is not None], [])
        result_dict = {"image_urls": all_image_urls}

        CrawlerHW.dict_save_to_file(result_dict, file_name)

        print(f"File save in {file_name}")

        return

    async def run(self):
        if len(sys.argv) < 2:
            print(CrawlerHW.ERROR_INPUT_MESSAGE)
            sys.exit()

        args_list = sys.argv[1:]

        async with httpx.AsyncClient() as client:
            start_run_time = time.time()

            if args_list[0] == "crawl":
                await self.crawl(client)

            elif args_list[0] == "push":
                await self.push(client, args_list[1], args_list[2])
            elif args_list[0] == "popular":
                await self.popular(client, args_list[1], args_list[2])

            elif args_list[0] == "keyword":
                await self.keyword(client, args_list[1], args_list[2], args_list[3])

            end_run_time = time.time()

            print(f"Run Time: {end_run_time - start_run_time}")


if __name__ == "__main__":
    asyncio.run(CrawlerHW().run())
