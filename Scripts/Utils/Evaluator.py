from copy import deepcopy
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.express as px
from scipy.stats import chi2_contingency
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef, roc_curve, auc
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from Utils.Ressources import Ressources

class PredStats(Ressources):
    def __init__(self, test_frame, predictions, test_name, model_name, train_name):
        self.load_paths()
        self.load_json()
        self.test_frame = test_frame
        self.predictions = predictions
        self.test_name = test_name
        self.model_name = model_name
        self.train_name = train_name
        self.prediction_path += self.test_name + '/'
        self.old_frame, self.old_frame_exists = self.load_frame()

    def load_frame(self):
        if not os.path.exists(self.prediction_path):
            os.makedirs(self.prediction_path)
        try:
            df = pd.read_csv(self.prediction_path + self.model_name + '_predictions.csv')
            return df, True
        except:
            return None, False

    def merge(self):
        if not self.old_frame_exists:
            self.test_frame[self.train_name] = self.predictions 
            new_frame = self.test_frame
        else:
            self.old_frame[self.train_name] = self.predictions 
            new_frame = self.old_frame
        self.save_frame(new_frame)

    def save_frame(self, df):
        df.to_csv(self.prediction_path + self.model_name + '_predictions.csv', index = False)

class Evaluator(Ressources):
    def __init__(self,
                model:str,
                test_frame:pd.DataFrame,
                train:str,
                test:str,
                task_type:str,
                y_true:np.array,
                y_probs:np.array,
                ):
        
        self.load_paths()
        self.load_json()
        self.all_results_path = self.result_path
        self.result_path += self.folder_format(model)
        self.model = model
        self.test_frame = test_frame
        self.train = train
        self.test = test
        self.task_type = task_type
        self.y_true = y_true
        self.y_pred = np.argmax(y_probs, axis=1) if self.model != 'XGBoost' else np.where(y_probs >= 0.5, 1, 0)
        self.y_probs = y_probs
        self.metric_dct = {
            "Model" : self.model,
            "Train" : self.train,
            "Test" : self.test,
            "Task" : self.task_type,
            'Time' : str(datetime.now())
            }
        self.create_folders()
        if not self.test_frame is None:
            predstats = PredStats(test_frame = self.test_frame, predictions = self.y_pred, test_name = self.test, model_name = self.model, train_name = self.train)
            predstats.merge()
        
        
    def create_folders(self):
        self.path = self.result_path
        if not os.path.exists(self.path + self.folder_format("train_" + self.train)):
            os.makedirs(self.path+ self.folder_format("train_" + self.train))   
        self.path += self.folder_format("train_" + self.train)
        if not os.path.exists(self.path + self.folder_format("test_" + self.test)):
            os.makedirs(self.path  + self.folder_format("test_" + self.test))
        self.path += self.folder_format("test_" + self.test)
        if not os.path.exists(self.path + self.folder_format("task_" + self.task_type)):
            os.makedirs(self.path  + self.folder_format("task_" + self.task_type))
        self.path += self.folder_format("task_" + self.task_type)
        
    
    def Metrics(self, y_true, y_pred):
        p = self.ChiSquare(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', labels=np.unique(y_pred))
        accuracy = accuracy_score(y_true, y_pred)
        matthews_corr = matthews_corrcoef(y_true, y_pred)
        return (accuracy, f1, precision, recall, matthews_corr, p)
        
    def ChiSquare(self,  y_true, y_pred):
        df = pd.DataFrame.from_dict({"True":y_true, "Prediction":y_pred})
        crosstab = pd.crosstab(df["True"], df["Prediction"])
        chi2_stat, p, dof, expected = chi2_contingency(crosstab)
        return p
    
    def ROC(self, y_true, y_probs):
        if np.unique(y_true).shape[0] > 2:
            return None, 0, 0
        if self.model != 'XGBoost':
            fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
        else:
            fpr, tpr, thresholds = roc_curve(y_true, y_probs)

        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        ix = np.argmax(np.sqrt(tpr * (1-fpr)))
        fig.add_traces(
            px.scatter(pd.DataFrame.from_dict({"False Positive Rate":[fpr[ix]], "True Positive Rate":[tpr[ix]]}),x="False Positive Rate", y="True Positive Rate").update_traces(marker_size=10, marker_color="red").data
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        return fig, thresholds[ix], auc(fpr, tpr)
    
    def save_probs(self, y_probs):
        with open(self.path + "probs.npy", 'wb') as f:
            np.save(f, y_probs)
            
    def save_metrics(self):
        with open(self.path + 'Metrics.json', 'w') as json_file:
            json.dump(self.metric_dct, json_file, indent=6)
            
    def save_visuals(self, fig, vis_name = "ConfusionMatrix"):
        path = self.path
        path += vis_name
        fig.write_image(path + ".png")
        fig.write_html(path + ".HTML")
        
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
    
    def fit(self, y_true, y_pred, y_probs):
        cm_fig = self.ConfusionMatrix(y_true, y_pred)
        roc_fig, best_cutoff, auc = self.ROC(y_true, y_probs)
        acc, f1, prec, recall, matthews_cor, chi2 = self.Metrics(y_true, y_pred)
        self.metric_dct["Accuracy"] = float(acc)
        self.metric_dct["F1"] = float(f1)
        self.metric_dct["Precision"] = float(prec)
        self.metric_dct["Recall"] = float(recall)
        self.metric_dct["Matthews Correlation"] = float(matthews_cor)
        self.metric_dct["Chi Square (p-value)"] = float(chi2)
        self.metric_dct["AUC"] = float(auc)
        self.metric_dct["Best Cutoff"] = float(best_cutoff)
        return cm_fig, roc_fig
    
 
    def evaluate(self):
        cm_fig, roc_fig = self.fit(self.y_true, self.y_pred, self.y_probs)
        self.save_visuals(cm_fig, "ConfusionMatrix")
        if roc_fig is not None:
            self.save_visuals(roc_fig, "ROC_Curve")
        self.save_probs(self.y_probs)
        self.save_metrics()
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
    x = np.array([0, 1, 2, 2, 2, 1])
    print(np.unique(x).shape[0])

    
    
    