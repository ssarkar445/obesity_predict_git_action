import optuna
from tunemodel import objective
import xgboost as xgb
from sklearn import metrics
from sklearn import multiclass


TRIALS=5

def train_xgb_model(train_data,FEATURES,ID,TARGET,fold):

    X_train = train_data.query("kfold!=@fold")[FEATURES].reset_index(drop=True)
    X_valid = train_data.query("kfold==@fold")[FEATURES].reset_index(drop=True)
    y_train = train_data.query("kfold!=@fold")[TARGET].reset_index(drop=True)
    y_valid = train_data.query("kfold==@fold")[TARGET].reset_index(drop=True)
    
    t_id = train_data[ID]
    v_id = train_data[ID]


    # study = optuna.create_study(direction="maximize")
    # study.optimize(lambda trial: objective(trial, X_train, X_valid, y_train, y_valid), n_trials=TRIALS)
    # params = study.best_params
    xgbmodel = multiclass.OneVsRestClassifier(estimator=xgb.XGBClassifier())
    xgbmodel.fit(X_train, y_train)
    test_pred = xgbmodel.predict(X_valid)
    print(f"Auc:{metrics.accuracy_score(y_valid,test_pred)}")
    return xgbmodel