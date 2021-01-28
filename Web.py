from flask import Flask, render_template, url_for, request
import numpy as np
import PreProcessing

app = Flask(__name__)

pp = PreProcessing

@app.route('/')
def info():
    return render_template('info.html')

@app.route('/index')
def index():
    #return "Index Page"
    return render_template('index.html')

@app.route('/',methods = ['POST'])
def result():
    grains = request.form['grains']
    vegetables = request.form['vegetables']
    fruits = request.form['fruits']
    protein = request.form['protein']
    grains = float(grains)
    vegetables = float(vegetables)
    fruits = float(fruits)
    protein = float(protein)
    predictedResult = pp.healthy_diet(grains,vegetables,fruits,protein)
    print('Experience is ',predictedResult)
    return render_template('result.html', PredictedResult = predictedResult,)
  
if __name__ == "__main__":
    #app.run()
    app.run(port=8000, debug=True)
