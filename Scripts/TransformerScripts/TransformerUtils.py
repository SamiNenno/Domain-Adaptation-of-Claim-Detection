import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
import emoji
import random
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from Dataloader.TrainingSetup import DataSetup
from Utils.Ressources import Ressources

random_seed = 2022
np.random.seed(random_seed)
random.seed(random_seed)

class TransformerConfigurations(DataSetup):
    def __init__(self, 
                model_language:str,
                model_size:str,
                datasetname:str,
                relabel:str):
        super().__init__()
        self.random_seed = 2022
        self.model_language = model_language.lower()
        self.model_size = model_size.lower()
        self.datasetname = datasetname.lower()
        self.relabel = relabel.lower()
        self.modelname = self.get_model_name(self.model_language, self.model_size)
        self.get_data()
        
    def get_model_name(self, language:str, size:str):
        transformer_dict = {
            "large":
                {
                    "multilingual" : "xlm-roberta-large",
                    "english" : "roberta-large",
                    "german" : "deepset/gbert-large"
                },
            "base" :
                {
                    "multilingual" : "xlm-roberta-base",
                    "english" : "roberta-base",
                    "german" : "bert-base-german-cased"
                }
        }  
        return transformer_dict[size][language] 

    def ValSplit(self, df, frac:float = 0.7):
        df_big = df.groupby("label").sample(frac=frac, random_state = self.random_seed)
        df_small = df.drop(df_big.index)
        return df_big.reset_index(drop=True), df_small.reset_index(drop=True) 

    def replace_emoji(self, df):
        df['text'] = df['text'].apply(lambda sentence: emoji.demojize(sentence))
        return df

    def get_data(self):
        self.train_frame, self.test_frame = self.choose_one(dataset = self.datasetname, split = True, relabel = self.relabel)
        self.train_frame = self.train_frame.rename(columns={"Text": "text", "Label": "label"})
        self.test_frame = self.test_frame.rename(columns={"Text": "text", "Label": "label"})
        if self.datasetname in ['checkthat2022', 'germeval']:
            self.train_frame, self.test_frame = self.replace_emoji(self.train_frame), self.replace_emoji(self.test_frame)
        self.num_labels = self.train_frame['label'].unique().shape[0]
        self.test_frame, self.val_frame = self.ValSplit(self.test_frame)

    def set_up_statistics(self):
        return {
            "modellanguage" : self.model_language,
            "modelsize" : self.model_size,
            "model" : self.modelname,
            "dataset" : self.datasetname,
            "task" : self.relabel,
            "labels" : self.num_labels,
        }

class TransformerDataLoader(TransformerConfigurations):
    def __init__(self,
                model_language:str,
                model_size:str,
                datasetname:str,
                relabel:str):
        super().__init__(model_language, model_size, datasetname, relabel)
        self.stats = self.set_up_statistics()
        self.tokenizer = AutoTokenizer.from_pretrained(self.stats["model"])
        self.turn_to_dataset()
    
    def turn_to_dataset(self):
        def encode(examples):
            outputs = self.tokenizer(examples['text'], truncation=True)
            return outputs
        self.trainset = Dataset.from_pandas(self.train_frame).map(encode, batched=True, new_fingerprint='A') #?
        self.valset = Dataset.from_pandas(self.val_frame).map(encode, batched=True, new_fingerprint='B') #?
        self.testset = Dataset.from_pandas(self.test_frame).map(encode, batched=True, new_fingerprint='C') #?

    def get_dict(self):
        return DatasetDict({'train':self.trainset, "val":self.valset, "test":self.testset})
    def get_stats(self):
        return self.stats

    

def get_data_and_stats(model_language:str,
                    model_size:str,
                    datasetname:str,
                    relabel:str):
    tloader = TransformerDataLoader(model_language, model_size, datasetname, relabel)
    return tloader.get_dict(), tloader.get_stats()

def get_path():
    ress = Ressources()
    ress.load_paths()
    ress.load_json()
    return ress.metadata_path, ress.model_path, ress.emission_path


