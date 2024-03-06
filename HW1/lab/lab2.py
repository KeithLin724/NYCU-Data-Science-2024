from bs4 import BeautifulSoup
import httpx


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


result = httpx.get("https://www.ptt.cc/bbs/Beauty/M.1672554775.A.108.html")

result = page_to_simple_dict(result.text)
print(result)
