import requests
import json
import os
import sys
from tqdm import tqdm
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
import plotly.express as px
import numpy as np
from datetime import datetime
from codecarbon import EmissionsTracker


PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from Dataloader.TrainingSetup import DataSetup



class ClaimbusterTester(DataSetup):
    def __init__(self) -> None:
        super().__init__()
        self.load_paths() 
        self.load_json()
        self.all_results_path = self.result_path
        self.pred_path = self.dataset_path + 'ClaimbusterPredictions/'
        self.result_path += 'ClaimbusterModel/'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        if not os.path.exists(self.pred_path):
            os.makedirs(self.pred_path)
        self.api_key = '5b53715de78a4f0f9cce290a4144d444'
        self.names = ['claimrank','checkthat2021', 'checkthat2019', 'checkthat2022', 'claimbuster', 'germeval','multifc']
        self.metric_dct = {
            "Model" : 'ClaimbusterTransformer',
            "Train" : 'claimbuster',
            "Task" : 'checkworthy',
            }

    def predict(self, input_claim):
        api_endpoint = f"https://idir.uta.edu/claimbuster/api/v2/score/text/sentences/{input_claim}"
        request_headers = {"x-api-key": self.api_key}
        api_response = requests.get(url=api_endpoint, headers=request_headers)
        return api_response.json()

    def apply_to_all_datasets(self):
        for idx, name in enumerate(self.names):
            print('----------------------')
            print(f'Dataset {idx+1}/{len(self.names)}: {name}')
            print('----------------------')
            tracker = EmissionsTracker(project_name=f"CLAIMBUSTER_API_{name}", output_dir=self.emission_path+'Baseline/')
            tracker.start()
            dct = {'sentence':[], 'groundtruth': [], 'prediction':[]}
            df = self.choose_one(dataset=name, split=False, relabel='checkworthy', augmentation = False)
            for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc = 'Predict'):
                dct['sentence'].append(row['Text'])
                dct['groundtruth'].append(row['Label'])
                try:
                    prediction = self.predict(row['Text'])['results'][0]['score']
                    dct['prediction'].append(prediction)
                except Exception as e:
                    dct['prediction'].append(str(e))
                if (idx+1) % 300 == 0:
                    self.save(pd.DataFrame.from_dict(dct), name)
            tracker.stop() 
            self.save(pd.DataFrame.from_dict(dct), name)

    def save(self, df, name):
        df.to_csv(self.pred_path + name + '.csv', index = False)

    def load(self, name):
        return pd.read_csv(self.pred_path + name + '.csv')

    def pred_from_prob(self, df, cutoff:float=.5):
        def helper(prob):
            if float(prob) >= cutoff:
                return 1
            else:
                return 0
        df['pred_label'] = df['prediction'].apply(lambda x: helper(x))
        return df

    def ChiSquare(self,  y_true, y_pred):
        df = pd.DataFrame.from_dict({"True":y_true, "Prediction":y_pred})
        crosstab = pd.crosstab(df["True"], df["Prediction"])
        chi2_stat, p, dof, expected = chi2_contingency(crosstab)
        return p

    def Metrics(self, y_true, y_pred):
        p = self.ChiSquare(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', labels=np.unique(y_pred))
        accuracy = accuracy_score(y_true, y_pred)
        matthews_corr = matthews_corrcoef(y_true, y_pred)
        return (accuracy, f1, precision, recall, matthews_corr, p)

    def ConfusionMatrix(self, y_true:np.array, y_pred:np.array):
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
        fig.update_layout({'xaxis_title':'Predictions'})
        fig.update_layout({'yaxis_title':'Ground Truth'})
        fig.update_layout(coloraxis_showscale=False)
        fig.update_layout(
            xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 1
            ),
            yaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 1
            )
        )
        return fig    

    def save_visuals(self, fig, path, vis_name = "ConfusionMatrix"):
        path += vis_name
        fig.write_image(path + ".png")
        fig.write_html(path + ".HTML")

    def save_metrics(self, path):
        with open(path + 'Metrics.json', 'w') as json_file:
            json.dump(self.metric_dct, json_file, indent=6)

    def evaluate(self, name):
        df = self.load(name)
        df = self.pred_from_prob(df)
        y_true = df['groundtruth'].to_numpy()
        y_pred = df['pred_label'].to_numpy()
        fig = self.ConfusionMatrix(y_true, y_pred)
        acc, f1, prec, recall, matthews_cor, chi2 = self.Metrics(y_true, y_pred)
        self.metric_dct["Test"] = name
        self.metric_dct["Time"] = str(datetime.now())
        self.metric_dct["Accuracy"] = float(acc)
        self.metric_dct["F1"] = float(f1)
        self.metric_dct["Precision"] = float(prec)
        self.metric_dct["Recall"] = float(recall)
        self.metric_dct["Matthews Correlation"] = float(matthews_cor)
        self.metric_dct["Chi Square (p-value)"] = float(chi2)
        self.metric_dct["AUC"] = None
        self.metric_dct["Best Cutoff"] = None
        path = self.result_path + name + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_visuals(fig, path)
        self.save_metrics(path)

    def evaluate_all(self):
        for name in tqdm(self.names, desc='Evaluate'):
            try:
                self.evaluate(name)
            except Exception as e:
                print(e)
        try:
            self.merge_results()
        except Exception as e:
            print(f"Merging Failed: {e}")

    def getListOfFiles(self, all_results_path):
        listOfFile = os.listdir(all_results_path)
        allFiles = list()
        for entry in listOfFile:
            fullPath = os.path.join(all_results_path, entry)
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.getListOfFiles(fullPath)
            else:
                allFiles.append(fullPath)        
        return [file for file in allFiles if "Metrics.json" in file]
    
    def merge_results(self):
        json_paths = self.getListOfFiles(self.all_results_path)
        dct_list = []
        for path in json_paths:
            with open(path, 'r') as f:
                dct = json.load(f)
            dct_list.append(dct)
        df = pd.DataFrame.from_dict(dct_list)
        df = df.dropna(subset=['Accuracy'])
        df["Chi (alpha = 0.1)"] = df['Chi Square (p-value)'].apply(lambda p: "Independent" if p > 0.1 else "Dependent")
        df["Chi (alpha = 0.05)"] = df['Chi Square (p-value)'].apply(lambda p: "Independent" if p > 0.05 else "Dependent")
        df["Chi (alpha = 0.01)"] = df['Chi Square (p-value)'].apply(lambda p: "Independent" if p > 0.01 else "Dependent")
        df = df.drop(['Chi Square (p-value)'], axis=1)
        df.to_csv(self.all_results_path + 'Results.csv', index = False)

if __name__ == '__main__':
    ct = ClaimbusterTester()
    ct.apply_to_all_datasets()
    ct.evaluate_all()
    
   
    