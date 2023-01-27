from zipfile import ZipFile
import pandas as pd
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from Dataloader.Dataset_Loader import DatasetLoader

class MultiFCLoader(DatasetLoader):
    def __init__(self):
        super().__init__()
        self.load_metadata("MultiFC")
        self.df = self.load_dataset()
        self.formatter()
        self.predefined_split = True
    
    def load_dataset(self):
        with ZipFile(self.dataset_path + 'multi_fc_publicdata.zip', 'r') as z:
            with z.open('train.tsv') as f:
                self.train = pd.read_csv(f, delimiter="\t", header=None)
            with z.open('dev.tsv') as f:
                val = pd.read_csv(f, delimiter="\t", header=None)
            with z.open('test.tsv') as f:
                test = pd.read_csv(f, delimiter="\t", header=None)
 
        self.test = pd.concat([val, test], ignore_index=True)
        return pd.concat([self.train, self.test], ignore_index=True)
       
    def formatter(self):
        self.df = self.df[[1]]
        self.df = self.df.rename(columns={1: "Text"})
        self.df = self.df[self.df['Text'].apply(lambda x: isinstance(x, (str, bytes)))]
        self.df['Label'] = 1 
        self.train = self.train[[1]]
        self.train = self.train.rename(columns={1: "Text"})
        self.train = self.train[self.train['Text'].apply(lambda x: isinstance(x, (str, bytes)))]
        self.train['Label'] = 1 
        self.test = self.test[[1]]
        self.test = self.test.rename(columns={1: "Text"})
        self.test = self.test[self.test['Text'].apply(lambda x: isinstance(x, (str, bytes)))]
        self.test['Label'] = 1 
        
    
if __name__ == '__main__':
    cl = MultiFCLoader()
    print(cl.df.shape)
   
    
