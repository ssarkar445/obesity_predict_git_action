import pandas as pd
from sklearn import preprocessing
import joblib

target_map = {'Obesity_Type_III': 0,
                'Obesity_Type_II': 1,
                'Normal_Weight': 2,
                'Obesity_Type_I': 3,
                'Insufficient_Weight': 4,
                'Overweight_Level_II': 5,
                'Overweight_Level_I': 6
                }


def feature_engineering(data,data_type='train'):
    if data_type=='train':
        data['BMI'] = data['Weight'] / (data['Height'] ** 2)
        data['Activity'] = data['FAF'] * data['TUE']
        categorical = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        categorical = [col for col in categorical if col not in('NObeyesdad')]
        data['NObeyesdad'] = data['NObeyesdad'].map(target_map).astype('int')
        train_data = pd.concat([data,pd.get_dummies(data[categorical],drop_first=True).astype(int)],axis=1)
        train_data = train_data.drop(categorical,axis=1)
        return train_data
    else:
        data['BMI'] = data['Weight'] / (data['Height'] ** 2)
        data['Activity'] = data['FAF'] * data['TUE']
        categorical = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        test_data = pd.concat([data,pd.get_dummies(data[categorical],drop_first=True).astype(int)],axis=1)
        test_data = test_data.drop(categorical,axis=1)
        return test_data

def scale_data(train,col_list):
    scaler = preprocessing.StandardScaler()
    train[col_list] = scaler.fit_transform(train[col_list])
    return scaler,train

if __name__=="__main__":
    pass

