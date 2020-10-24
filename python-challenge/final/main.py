
"""
These are the URLs that will give you remote jobs for the word 'python'

https://stackoverflow.com/jobs?r=true&q=python
https://weworkremotely.com/remote-jobs/search?term=python
https://remoteok.io/remote-dev+python-jobs

Good luck!
"""

import requests
from flask import Flask, render_template, request
from bs4 import BeautifulSoup

app=Flask("remote_job")

@app.route("/remote_job")
def remote_job():
    word=request.args.get('word')
    return render_template("remote_job.html")

@app.route("/")
def home():
    return render_template("home.html")

app.run(host="localhost")