from flask import Flask,request,jsonify
from flask_cors import CORS 
import numpy as np
import pickle

model = pickle.load(open('savedmodel.sav','rb'))

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    temp = float(request.form['temp'])
    pH = float(request.form['pH'])
    conductivity = float(request.form.get('conductivity'))
    tds = float(request.form.get('tds'))
    bod = float(request.form.get('bod'))
    nitrate = float(request.form.get('nitrate'))
    flouride = float(request.form.get('flouride'))
    arsenic = float(request.form.get('arsenic'))

    #result={'temp':temp,'pH':pH,'conductivity':conductivity,'tds':tds,'bod':bod,'nitrate':nitrate,'flouride':flouride,'arsenic':arsenic}
   # return jsonify(result)

    input_query = np.array([[temp,pH,conductivity,tds,bod,nitrate,flouride,arsenic]])
    result = model.predict(input_query)[0]

    return jsonify({'Status':str(result)})


if __name__ == '__main__':
    app.run(debug=True)