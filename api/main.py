from flask import Flask
import json
import numpy as np
import pickle

app = Flask(__name__)

with open("./api/database.pkl", "rb") as file:
    database = pickle.load(file)

@app.route("/")
def home():
    return "HELLO WORLD"

#Test
@app.route("/test")
def test():
    return str(database.keys())