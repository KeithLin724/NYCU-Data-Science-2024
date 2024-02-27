import httpx
import bs4

url = "https://www.ptt.cc/bbs/Beauty/M.1672503968.A.5B5.html"
header = {"cookie": "over18=1"}

result = httpx.get(url=url, headers=header)

with open("sample.html", mode="w", encoding="utf-8") as f:
    f.writelines(str(result.text))

print(result.text)
