from flask import Flask, render_template,request
import pickle
import numpy as np

app=Flask(__name__)

with open('house_price_prediction.pkl','rb')as f:
        model=pickle.load(f);

@app.route('/')
def home():
        return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
        features=[
            float(request.form['CRIM']),
            float(request.form['ZN']),
            float(request.form['INDUS']),
            float(request.form['CHAS']),
            float(request.form['NOX']),
            float(request.form['RM']),
            float(request.form['AGE']),
            float(request.form['DIS']),
            float(request.form['RAD']),
            float(request.form['TAX']),      
            float(request.form['PTRATIO']),
            float(request.form['B']),
            float(request.form['LSTAT'])
        ]
        feature_array=np.array([features])

        prediction=model.predict(feature_array)
        output=round(prediction[0],2)
        prediction_text = f"PREDICTED PRICE: ${output}"
        



        return render_template('index.html',prediction=prediction_text)

if __name__ == "__main__":
        app.run(debug=True)