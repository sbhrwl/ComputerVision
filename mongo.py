from flask import Flask
from flask import render_template,request
import pymongo
from flask_pymongo import PyMongo
#from testing import query
from pymongo import MongoClient
from flask import jsonify
import json
from bson.json_util import dumps
import json, os, signal
from colorama import Fore

app = Flask(__name__,template_folder='templates')
"""app.config["MONGO_URI"] = "mongodb://localhost:27017/test"
mongo = PyMongo(app)
db = mongo.test"""

client = MongoClient()
#print(client.list_database_names())
db= client.test

"""@app.route("/")
def home_page():
    online_users = mongo.db.users.find({"car_ID": True})
    return render_template("index.html",online_users=online_users)"""

@app.route('/')
def main_page():
    return render_template('index.html')

"""@app.route("/", methods = ['POST'])
def shakunt():
    #global text
    text = request.form["u"]
    #return text
    output = query(text)
    return output
    #return render_template(text=text)"""

@app.route("/",methods = ['POST'])
def final():
    text = request.form['carname']
    print(text)
    online_users = db.test_info.find({"CarName": text},{'wheelbase':1})
    print(online_users)
    print(type(online_users))
    for i in online_users:
        print(i)
        print(type(i))
        output = i.get('wheelbase')
        final_output = dumps(output)
        print(final_output)
    #return jsonify(online_users)
    print(Fore.GREEN + "Your program ran successfully")
    #return final_output
    return render_template('index.html', final_output = final_output)

    #return render_template("index.html", online_users=online_users)


#print(text)
"""def get_value():
    local_text = request.form["u"]
    print(local_text)"""


if __name__ == "__main__":
    app.run(debug=True, port = 8000)

