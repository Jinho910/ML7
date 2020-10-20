from flask import Flask, render_template, request, redirect
import requests
from bs4 import BeautifulSoup

base_url = "http://hn.algolia.com/api/v1"

# This URL gets the newest stories.
new = f"{base_url}/search_by_date?tags=story"

# This URL gets the most popular stories
popular = f"{base_url}/search?tags=story"


# This function makes the URL to get the detail of a storie by id.
# Heres the documentation: https://hn.algolia.com/api

def make_detail_url(id):
    return f"{base_url}/items/{id}"


def get_story():
    data = requests.get(popular).json()


db = {}
app = Flask("DayNine")


@app.route("/")
def home():
  order="popular"
  data=requests.get(popular).json()
  # print(data['hits'])
  for story in data['hits']:
      print(story['title'])
  # return render_template("index.html",order_by=order, data=data.get['hits'])
  return render_template("index.html",order_by=order, stories=data.get('hits'))



app.run(host="localhost")
