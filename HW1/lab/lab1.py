from bs4 import BeautifulSoup
import httpx

from rich import print
import random as rd

USER_AGENT_LIST = [
    # Opera
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36 OPR/26.0.1656.60",
    "Opera/8.0 (Windows NT 5.1; U; en)",
    "Mozilla/5.0 (Windows NT 5.1; U; en; rv:1.8.1) Gecko/20061208 Firefox/2.0.0 Opera 9.50",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; en) Opera 9.50",
    # Firefox
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0",
    "Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10",
    # Safari
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.57.2 (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2",
    # chrome
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 Safari/534.16",
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
    "Mozilla/5.0 (iPhone; CPU iPhone OS 13_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 SP-engine/2.14.0 main%2F1.0 baiduboxapp/11.18.0.16 (Baidu; P2 13.3.1) NABar/0.0 ",
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


def user_agent():
    temp = {"User-Agent": rd.choice(USER_AGENT_LIST)}
    print(temp)
    return temp


# def page_to_simple_dict(html_str: str) -> dict:
#     """
#     format like
#     '作者': 'ReiKuromiya (ReiKuromiya)',
#     '標題': '[正妹] 周子瑜',
#     '時間': 'Sun Jan  1 00:26:06 2023',
#     'Year': '2023',
#     'Month': 'Jan',
#     'Date': '',
#     'Week': 'Sun',
#     'Time': '00:26:06',
#     'Body': [(..., ...),  ...]}
#     """

#     def to_detail_date(date_str: str) -> dict:
#         detail_date = date_str.split(" ")
#         return {
#             "Year": detail_date[-1],
#             "Month": detail_date[1],
#             "Date": detail_date[2],
#             "Week": detail_date[0],
#             "Time": detail_date[-2],
#         }

#     # header = {"cookie": "over18=1"}
#     # result = httpx.get(url=url, headers=header)

#     soup = BeautifulSoup(html_str, "html.parser")

#     # get main data
#     body_data = soup.find("div", class_="bbs-screen bbs-content", id="main-content")

#     # get header data
#     header_data = body_data.find_all("div", class_="article-metaline")

#     tab_list = [
#         str(line.find("span", class_="article-meta-tag").string) for line in header_data
#     ]

#     value_list = [
#         str(line.find("span", class_="article-meta-value").string)
#         for line in header_data
#     ]

#     header_dict = dict(zip(tab_list, value_list))

#     # image link lists
#     link_image = body_data.find_all("a")
#     link_image = [link.string for link in link_image]

#     # image src
#     images_list = body_data.find_all("div", class_="richcontent")
#     images_list = [item.find("img").get("src") for item in images_list]

#     content_images = list(zip(link_image, images_list))

#     page_data = (
#         header_dict | to_detail_date(header_dict["時間"]) | {"Body": content_images}
#     )

#     return page_data


def page_to_simple_dict(html_str: str) -> dict:
    """
    format like
    '作者': 'ReiKuromiya (ReiKuromiya)',
    '標題': '[正妹] 周子瑜',
    '時間': 'Sun Jan  1 00:26:06 2023',
    'Year': '2023',
    'Month': 'Jan',
    'Date': '',
    'Week': 'Sun',
    'Time': '00:26:06',
    'Body': [(..., ...),  ...]}
    """

    def to_detail_date(date_str: str) -> dict:
        detail_date = date_str.split(" ")
        detail_date.remove("")
        return {
            "Year": detail_date[-1],
            "Month": detail_date[1],
            "Date": detail_date[2],
            "Week": detail_date[0],
            "Time": detail_date[-2],
        }

    # header = {"cookie": "over18=1"}
    # result = httpx.get(url=url, headers=header)

    soup = BeautifulSoup(html_str, "html.parser")

    # get main data
    body_data = soup.find("div", class_="bbs-screen bbs-content", id="main-content")

    # get header data
    header_data = body_data.find_all("div", class_="article-metaline")

    tab_list = [
        str(line.find("span", class_="article-meta-tag").string) for line in header_data
    ]

    value_list = [
        str(line.find("span", class_="article-meta-value").string)
        for line in header_data
    ]

    header_dict = dict(zip(tab_list, value_list))

    # image src
    images_list = body_data.find_all("div", class_="richcontent")

    # base in how long in image_list
    images_list = [
        item.get("src") for image in images_list if (item := image.find("img"))
    ]
    # print(images_list)

    # image link lists
    link_image = body_data.find_all("a")
    # {".png" , ".jpg" , "jpeg" ,".gif"} in (link_str := str(link.string))
    link_image = [
        link_str
        for link in link_image
        if any(
            substring in (link_str := str(link.string))
            for substring in [".png", ".jpg", "jpeg", ".gif"]
        )
    ]

    # content_images = list(zip(link_image[:len(images_list)], images_list))

    page_data = (
        header_dict
        | to_detail_date(header_dict["時間"])
        | {"image_catch_list": images_list, "image_link": link_image}
    )

    return page_data


def recommend_page_to_simple_dict(html_str: str) -> dict:

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


with httpx.Client() as client:
    header = {"cookie": "over18=1"}

    test_1 = client.get(
        "https://www.ptt.cc/bbs/Beauty/M.1672503968.A.5B5.html",
        headers=header
        | {
            "User-Agent": "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)"
        },
    )
    test = page_to_simple_dict(html_str=test_1.text)
    print(test)

    test_1 = client.get(
        "https://www.ptt.cc/bbs/Beauty/M.1708859265.A.A7D.html",
        headers=header | user_agent(),
    )
    test = page_to_simple_dict(html_str=test_1.text)
    print(test)

    test_2 = client.get(
        "https://www.ptt.cc/bbs/Beauty/index4000.html", headers=header | user_agent()
    )

    test = recommend_page_to_simple_dict(html_str=test_2.text)
    print(test)

    test_2 = client.get(
        "https://www.ptt.cc/bbs/Beauty/index3662.html", headers=header | user_agent()
    )

    test = recommend_page_to_simple_dict(html_str=test_2.text)
    print(test)
