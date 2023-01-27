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

class GermEvalLoader(DatasetLoader):
    def __init__(self):
        super().__init__()
        self.load_metadata("GermEval")
        self.df = self.load_dataset()
        self.formatter()
        self.predefined_split = True
    
    def load_dataset(self):
        with ZipFile(self.dataset_path + 'germeval2021toxic SharedTask main Data%20Sets.zip', 'r') as z:
            with z.open('Data Sets/GermEval21_TestData.csv') as f:
                self.test = pd.read_csv(f)
            with z.open('Data Sets/GermEval21_TrainData.csv') as f:
                self.train = pd.read_csv(f)
        return pd.concat([self.train, self.test], ignore_index=True)
    
    def formatter(self):
        self.df = self.df[["comment_text", "Sub3_FactClaiming"]]
        self.df = self.df.rename(columns={"comment_text": "Text", "Sub3_FactClaiming": "Label"})
        self.test = self.test[["comment_text", "Sub3_FactClaiming"]]
        self.test = self.test.rename(columns={"comment_text": "Text", "Sub3_FactClaiming": "Label"})
        self.train = self.train[["comment_text", "Sub3_FactClaiming"]]
        self.train = self.train.rename(columns={"comment_text": "Text", "Sub3_FactClaiming": "Label"})

    
if __name__ == '__main__':
    gl = GermEvalLoader()
    print(gl.get_data())
   
    
    
