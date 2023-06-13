import requests
from bs4 import BeautifulSoup
import re

def scrape_and_save(url):
    # 发送HTTP请求并获得响应内容
    response = requests.get(url)

    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(response.text, "html.parser")

    # 获取所有段落（<p>）元素并遍历它们
    paragraphs = soup.find_all("p")
    with open("output.txt", "a", encoding="utf-8") as f:
        for p in paragraphs:
            # 移除HTML标签、空格和空行，并保留汉字和标点符号
            clean_text = re.sub(r"\s+", "", p.get_text())
            clean_text = re.sub(r"[^。\w\s\u4e00-\u9fa5]+", "", clean_text)

            # 将文本保存到文件中
            if clean_text:
                f.write(clean_text)

# 爬取搜狐新闻财经板块的第一个页面
url = "https://www.sohu.com/a/670599499_130887?scm=1103.plate:355:0.0.1_1.0&spm=smpc.channel_218.block4_113_ugzL7M_1_fd.1.1682577001315PcF9ZWr_704"
scrape_and_save(url)

