from imghdr import tests
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import json
import sys
from tqdm import tqdm
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from Utils.Ressources import Ressources
from Utils.Evaluator import Evaluator

class XGTester(Ressources):
    def __init__(self) -> None:
        self.load_paths()
        self.load_json()
        self.random_seed = 2022
        

    def get_hypparam(self, dataset:str):
        with open(f'{self.xgboost_path}{dataset}/Hypparam.json', 'r') as f:
            hypparam = json.load(f)
        return hypparam
    
    def get_test_path(self, dataset:str):
        return f'{self.embeddings_path}not_augmented/{dataset}_test.csv'
    
    def get_train_path(self, dataset:str):
        return f'{self.embeddings_path}augmented/{dataset}_train.csv'
    
    def ValSplit(self, df, frac:float = 0.9):
        df_big = df.groupby("Label").sample(frac=frac, random_state = self.random_seed)
        df_small = df.drop(df_big.index)
        return df_big.reset_index(drop=True), df_small.reset_index(drop=True) 
    
    def test(self, model, trainset:str, testset:str):
        df1 = pd.read_csv(self.get_test_path(testset))
        if trainset == testset:
            test_frame =  df1
        elif testset == 'multifc':
            test_frame =  df1
            test_frame = test_frame.fillna(0)
        else:
            df2 = pd.read_csv(self.get_train_path(testset))
            test_frame = pd.concat([df1, df2])
            
        X_test = xgb.DMatrix(test_frame.iloc[:,2:].to_numpy())
        y_true = test_frame.iloc[:,1].to_numpy().reshape(-1)
        y_probs = model.predict(X_test)
        return test_frame, y_true, y_probs
    
    def train(self, trainset:str):
        hypparam = self.get_hypparam(trainset)
        train_frame = pd.read_csv(self.get_train_path(trainset))
        train_frame, val_frame = self.ValSplit(train_frame)
        y_train = train_frame.iloc[:,1].to_numpy().reshape(-1)
        y_eval = val_frame.iloc[:,1].to_numpy().reshape(-1)
        X_train = xgb.DMatrix(train_frame.iloc[:,2:].to_numpy(), label = y_train)
        X_val = xgb.DMatrix(val_frame.iloc[:,2:].to_numpy(), label = y_eval)
        evallist = [(X_val, 'eval'), (X_train, 'train')]
        num_round = 5000
        model = xgb.train(hypparam, X_train, num_round, evallist)
        return model

    def evaluate(self, model_name, test_frame, train, test, task_type, y_true, y_probs):
        if train == test:
            test_frame = None
        evaluator = Evaluator(model = model_name,
                    test_frame=test_frame,
                    train = train,
                    test = test,
                    task_type = task_type,
                    y_true = y_true, 
                    y_probs = y_probs,
                    )
        evaluator.evaluate()

    def fit(self, train_name:str, test_name:str):
        task_type = 'checkworthy'
        model = self.train(train_name)
        test_frame, y_true, y_probs = self.test(model=model, trainset=train_name, testset=test_name)
        self.evaluate('XGBoost', test_frame, train_name, test_name, task_type, y_true, y_probs)
    
    def fit_all(self):
        names = ['claimrank', 'claimbuster','checkthat2021', 'checkthat2019', 'checkthat2022', 'germeval']
        for train_name in tqdm(names, desc='Iterate trainsets', total=len(names), position=0):
            for test_name in tqdm(names + ['multifc'], desc='Iterate testsets', total=len(names)+1, position=1, leave=False):
                try:
                    self.fit(train_name, test_name)
                except Exception as e:
                    print(e)

if __name__ == '__main__':
    tester = XGTester()
    tester.fit_all()
        

    

