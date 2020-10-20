import requests
from bs4 import BeautifulSoup
# import json


data=requests.get("http://hn.algolia.com/api/v1/search?tags=story").json()

for d in data.get('hits'):
    print(d)