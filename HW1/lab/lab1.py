from bs4 import BeautifulSoup
import httpx

from rich import print


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

    # image link lists
    link_image = body_data.find_all("a")
    link_image = [link.string for link in link_image]

    # image src
    images_list = body_data.find_all("div", class_="richcontent")
    images_list = [item.find("img").get("src") for item in images_list]

    content_images = list(zip(link_image, images_list))

    page_data = (
        header_dict | to_detail_date(header_dict["時間"]) | {"Body": content_images}
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
        "https://www.ptt.cc/bbs/Beauty/M.1672503968.A.5B5.html", headers=header
    )
    test = page_to_simple_dict(html_str=test_1.text)
    print(test)

    test_2 = client.get("https://www.ptt.cc/bbs/Beauty/index4000.html", headers=header)

    test = recommend_page_to_simple_dict(html_str=test_2.text)
    print(test)
