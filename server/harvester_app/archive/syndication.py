from bs4 import BeautifulSoup as sup
from lxml import html
import requests


# getting SEC press releases articles 
btc_news_feed = 'https://news.bitcoin.com/feed/'
uri = requests.get("https://www.sec.gov/news/pressreleases.rss")
uri_fed = 'https://www.federalreserve.gov/feeds/press_all.xml'
print(uri)
links = []

raw = sup(uri.content, "xml")
ents = raw.find_all("item")
n=0
for e in ents:
    n+=1
    links.append(e.link.text)
    print(n)
    print(e.title.text)

page = requests.get(links[0])
print(links[0])

ciorba = sup(page.content, "html.parser")
job_elements = ciorba.find("div", class_="article-body")
print(ciorba.prettify())
print(job_elements.text)
'''
job_elements = ciorba.find_all("div", class_="article-body")


for job_element in job_elements:
    location_element = job_element.find_all("p")
    for p in location_element:
          print(p.text, '\n\n')
'''