from cProfile import run
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
app = Flask(__name__)

#load the model
model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index1.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("/index1.html",prediction_text="The flower species is {}".format(prediction) )

if __name__=="__main__":
    app.run(debug=True)