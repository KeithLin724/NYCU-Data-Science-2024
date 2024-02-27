from bs4 import BeautifulSoup
import httpx


def page_to_simple_dict(url: str):
    header = {"cookie": "over18=1"}
    result = httpx.get(url=url, headers=header)

    soup = BeautifulSoup(result.text, "html.parser")

    # get main data
    body_data = soup.find("div", class_="bbs-screen bbs-content", id="main-content")

    # header
    header_data = body_data.find_all("div", class_="article-metaline")

    header_dict = {
        str(line.find("span", class_="article-meta-tag").string): str(
            line.find("span", class_="article-meta-value").string
        )
        for line in header_data
    }

    # image link lists
    link_image = body_data.find_all("a")
    link_image = [link.string for link in link_image]

    # image src
    images_list = body_data.find_all("div", class_="richcontent")
    images_list = [item.find("img").get("src") for item in images_list]

    content_images = list(zip(link_image, images_list))

    page_data = header_dict | {"body": content_images}

    return page_data


# test = page_to_simple_dict(url="https://www.ptt.cc/bbs/Beauty/M.1672503968.A.5B5.html")
# print(test)
