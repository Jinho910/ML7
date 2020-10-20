
from flask import Flask, request, redirect, render_template

app=Flask("jbo")

@app.route("/")
def home():
    return render_template('potato.html')

@app.route('/report')
def report():
    word=request.args.get('word_to_find')
    return render_template('report.html',word=word)
    # return f"hello this is {var1}"
    # print(request.get('word'))
    # return "Report"

app.run(host='localhost')

#https://hn.algolia.com/?sort=byPopularity
#http://hn.algolia.com/api/v1/search?query=