import pandas as pd
from pathlib import Path
import zipfile
from sklearn import model_selection



with zipfile.ZipFile('RawData/train.csv.zip','r') as file:
    file.extractall("RawData")

df = pd.read_csv('RawData/train.csv')

skf = model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=2024)
df = df.sample(frac=1).reset_index(drop=True)
df['kfold'] = -99

for fold,(tidx,vidx) in enumerate(skf.split(X=df,y=df['NObeyesdad'])):
    df.loc[vidx,'kfold'] = fold

df.to_csv('./DataFold/train_folds.csv',index=False)