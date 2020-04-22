# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:35:14 2020

@author: PawanKumar
"""

from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
#Load the pickle file of model
model = pickle.load(open('random_forest.pkl' ,'rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    var1 = request.form['num1']
    var2 = request.form['num2']
    var3 = request.form['num3']
    var4 = request.form['num4']
    var5 = request.form['num5']
    var6 = request.form['num6']
    var7 = request.form['num7']
    var_array = np.array([var1, var2, var3, var4, var5, var6, var7]).reshape(1,-1)
    
#    df = pd.read_csv('/Users/pawanKumar/ML Notes/AQI Project/Data/Real-Data/Real_2014.csv')
    my_prediction = model.predict(var_array)
    my_prediction = round(np.float64(my_prediction),2)
#    my_prediction=my_prediction.tolist()
    return render_template('result.html', prediction = my_prediction)

if __name__== '__main__':
    app.run(debug = True,use_reloader=False)


