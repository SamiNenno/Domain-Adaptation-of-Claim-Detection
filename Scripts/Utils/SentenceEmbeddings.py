from dataclasses import dataclass
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from codecarbon import EmissionsTracker
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from Dataloader.TrainingSetup import DataSetup

class SentenceEmbedding(DataSetup):
    def __init__(self, dataset:str, augmentation:bool) -> None:
        super().__init__()
        self.dataset = dataset
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1', device = 'cuda') #https://www.sbert.net/examples/training/multilingual/README.html
        if self.dataset != 'multifc':
            self.train, self.test = self.choose_one(dataset = dataset, split = True, relabel = "checkworthy", augmentation = augmentation)
            self.train_path, self.test_path = self.setup_folder(self.dataset_path, augmentation, dataset)

    def embed_multifc(self):
        self.df = self.choose_one(dataset = 'multifc', split = False, relabel = "checkworthy", augmentation = False)
        self.train_path, self.test_path = self.setup_folder(self.dataset_path, False, 'multifc')
        tracker = EmissionsTracker(project_name="SentenceEmbedding", output_dir=self.emission_path)
        tracker.start()
        embeddings = self.encode(self.df)
        tracker.stop()
        self.save(pd.concat([self.df, embeddings], axis=1), train = False)

    def setup_folder(self, path:str, augmentation:bool, dataset:str):
        path += 'SentenceEmbeddings/'
        if not os.path.exists(path):
            os.makedirs(path) 
        if augmentation:
            path += f'augmented/'
        else:
            path += f'not_augmented/'
        if not os.path.exists(path):
            os.makedirs(path) 
        return path + dataset + '_train.csv', path + dataset + '_test.csv'

    def encode(self, df:pd.DataFrame):
        sentences = df['Text'].to_list()
        return pd.DataFrame(self.model.encode(sentences))

    def embed(self):
        #self.df = self.df.iloc[:1000,:]
        tracker = EmissionsTracker(project_name="SentenceEmbedding", output_dir=self.emission_path)
        tracker.start()
        train_embeddings = self.encode(self.train)
        test_embeddings = self.encode(self.test)
        tracker.stop()
        self.save(pd.concat([self.train, train_embeddings], axis=1), train = True)
        self.save(pd.concat([self.test, test_embeddings], axis=1), train = False)

    def save(self, df:pd.DataFrame, train:bool):
        path = self.train_path if train else self.test_path
        df.to_csv(path, index = False)

if __name__ == '__main__':
    #for dataset in ['claimbuster', 'checkthat2019', 'checkthat2021', 'checkthat2022', 'germeval']:
    #    se = SentenceEmbedding(dataset, False)
    #    se.embed()'''
    se = SentenceEmbedding('claimrank', False)
    se.embed()
    