from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from model import imagePredict

import os

app = Flask(__name__)

@app.route('/')
def home() :
    return render_template("home.html")

@app.route('/upload', methods = ['GET', 'POST'])
def view():
    if request.method == 'POST' :
        f = request.files['file']
        f.save('./static/uploads/' + secure_filename(f.filename))
        result = imagePredict('./static/uploads/' + secure_filename(f.filename))
        print(result)
        return render_template('result.html', image_file = './uploads/' + secure_filename(f.filename), \
            pre1 = result[0], pre2 = result[1], rec = result[2])

# @app.route('/result', methods = ['GET', 'POST'])
# def view():
#     return render_template('result.html', image_file = './uploads/' + secure_filename(f.filename))

if __name__ == '__main__' :
    app.run('127.0.0.1', 5000, debug = True)