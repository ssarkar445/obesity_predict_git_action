from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
from preprocess_data import feature_engineering
import joblib

app = Flask('obesity')

@app.route('/predict', methods=['POST'])

def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file part'})

    if file:
        try:
            df = pd.read_csv(file)
            df = feature_engineering(df,data_type='test')
            scale_col = ['Age','FCVC','NCP','CH2O','BMI','Activity']
            scaler = joblib.load('scaler/scaler.pkl') 
            df[scale_col] = scaler.transform(df[scale_col])
            FEATURES = joblib.load('features/features.pkl') 
            df = df[FEATURES]

            fold=5
            pred=[]
            for f in range(fold):
                model = joblib.load(f'model/moded_fold_{f}.pkl') 
                p = model.predict(df)
                pred.append(p)
            final_pred = np.mean(pred,axis=0)
            return jsonify({'predictions': final_pred.tolist()})
        except Exception as e:
            return jsonify({'error': str(e)})
        
if __name__=="__main__":
    app.run(debug=True)