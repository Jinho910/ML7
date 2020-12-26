"""
These are the URLs that will give you remote jobs for the word 'python'

https://stackoverflow.com/jobs?r=true&q=python
https://weworkremotely.com/remote-jobs/search?term=python
https://remoteok.io/remote-dev+python-jobs

Good luck!
"""

import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, send_file
from stack_overflow import get_job as so_job
from wework import get_job as ww_job
from remoteok import get_job as rw_job
from save import save_to_file


db = {}

app = Flask("remote_job")


@app.route("/remote_job")
def remote_job():
    word = request.args.get('word')
    word = word.lower()
    if word in db:
        jobs = db.get(word)
    else:
        so_jobs=so_job(word)
        ww_jobs=ww_job(word)
        rw_jobs=rw_job(word)
        jobs=so_jobs+ww_jobs+rw_jobs
        db[word] = jobs

    return render_template("remote_job.html", word=word, job_num=len(jobs), jobs=jobs)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/export")
def export():
    # try:
        word = request.args.get('word')
    # TODO c++ 일때 버그 수정할것
        # if not word:
        #     raise Exception()
        word = word.lower()
        jobs = db.get(word)
        # if not jobs:
        #     raise Exception()
        #
        save_to_file(f'{word}.csv',jobs)
        return send_file(f'{word}.csv', as_attachment=True)
        # return "c++"
    # except:
    #     # print("exception occured")
    #     return redirect("/")


app.run(host="localhost")
