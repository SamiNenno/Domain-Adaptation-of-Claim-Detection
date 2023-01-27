import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from Dataloader.DataCaller import DataCaller
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from scipy.spatial import distance
from scipy.special import softmax
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

class VanillaTransformer():
    def __init__(self, df, epochs = 5, batch_size=32):
        self.df = df.copy()
        self.df['DATASET_ID'] = self.df.index
        self.df = self.df.rename(columns={"Text":"Text",  "Label": "CHECKWORTHINESS_LABEL", "DATA_LABEL" :"label", "DATA_NAME": "DATA_NAME",  "DATASET_ID":"DATASET_ID"})
        self.df = self.df.sample(frac=1, random_state=2022)
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.num_labels = self.df['label'].unique().shape[0]
        self.training_args = TrainingArguments(output_dir="test_trainer", do_eval=False, per_device_train_batch_size=batch_size, num_train_epochs= epochs)
            
    def tokenize_function(self, examples):
            return self.tokenizer(examples["Text"], padding="max_length", truncation=True)
        
    def build_dataset(self, train_frame, test_frame):
        train, test = Dataset.from_pandas(train_frame), Dataset.from_pandas(test_frame)
        return train.map(self.tokenize_function, batched=True), test.map(self.tokenize_function, batched=True)      
        
    def train(self):
        n_splits = 5
        kf = KFold(n_splits=n_splits)
        for train, test in tqdm(kf.split(self.df), desc=f'{n_splits}-fold Cross-Validation', total=n_splits):
            torch.cuda.empty_cache()
            model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=self.num_labels)
            train_frame, test_frame = self.df.iloc[train,:].copy(), self.df.iloc[test,:].copy()
            traindata, testdata = self.build_dataset(train_frame, test_frame)
            trainer = Trainer(model=model,args=self.training_args,train_dataset=traindata)
            trainer.train()
            probs = pd.DataFrame(softmax(trainer.predict(testdata).predictions, axis=1))
            probs['DATASET_ID'] = self.df.iloc[test,:]['DATASET_ID'].to_list()
            yield probs
            
    def predict_proba(self):
        probs = pd.concat([df for df in self.train()]).sort_values(by=['DATASET_ID'])
        return probs.iloc[:,:-1].to_numpy()
    
    def fit(self):
        probs = self.predict_proba()
        preds = np.argmax(probs, axis=1)
        self.df['PREDICTION_LABEL'] = preds
        return self.df

class DatasetPrediction(DataCaller):
    def __init__(self):
        super().__init__()
        self.load_paths()
        self.load_json()
        self.select(["all"])
        self.df = self.load_datasets()
        self.PATH = f"{self.statistics_path}DatasetPredictions.csv"
        
    def load_datasets(self):
        df_list = []
        self.idx_to_label = {}
        for idx, data_object in tqdm(enumerate(self.get_all_data()), desc="Load Data"):
            df = data_object.get_data()
            name = data_object.get_name()
            df['DATA_LABEL'] = idx
            df['DATA_NAME'] = name
            self.idx_to_label[idx] = name
            df_list.append(df)
        return pd.concat(df_list)
    
    def fit(self):
        vt = VanillaTransformer(self.df)
        self.df = vt.fit()
        self.df['PREDICTION_NAME'] = self.df['PREDICTION_LABEL'].apply(lambda x : self.idx_to_label[x])
        self.df.to_csv(self.PATH, index = False)
        return self.df
    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    dp = DatasetPrediction()
    print(dp.fit())