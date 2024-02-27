import asyncio
import httpx
import bs4

MAIN_URL = "https://www.ptt.cc/bbs/Beauty/M.1672503968.A.5B5.html"
START_URL = "https://www.ptt.cc/bbs/Beauty/index3662.html"


class CrawlerHW:
    def __init__(self) -> None:
        pass

    async def run(self):
        print("Hello")


if __name__ == "__main__":
    asyncio.run(CrawlerHW().run())
