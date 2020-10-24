import requests
from bs4 import BeautifulSoup
URL="https://stackoverflow.com/jobs?r=true&q=python"
html=requests.get(URL)
soup=BeautifulSoup(html.text, 'html.parser')
soup=BeautifulSoup(open('so.html', encoding='utf-8') ,'html.parser')
pagination=soup.find('div', {"class":"s-pagination"})
pages=pagination.find_all("a")
print(len(pages))
last_page=len(pages)-1

for p in range(last_page):
    page_num=p+1
    url=f"https://stackoverflow.com/jobs?q=python&r=true&pg={page_num}"
