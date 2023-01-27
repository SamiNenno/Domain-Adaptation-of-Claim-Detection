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

class Checkthat2022Loader(DatasetLoader):
    def __init__(self):
        super().__init__()
        self.load_metadata("Checkthat2022")
        self.df = self.load_dataset()
        self.formatter()
        self.predefined_split = True
    
    def load_dataset(self):
        with ZipFile(self.dataset_path + 'CT22_english_1A_checkworthy.zip', 'r') as z:
            with z.open('CT22_english_1A_checkworthy_dev_test.tsv') as f:
                test = pd.read_csv(f, delimiter="\t")
            with z.open('CT22_english_1A_checkworthy_dev.tsv') as f:
                val = pd.read_csv(f, delimiter="\t")
            with z.open('CT22_english_1A_checkworthy_train.tsv') as f:
                self.train = pd.read_csv(f, delimiter="\t")
        self.test = pd.concat([test, val], ignore_index=True)     
        return pd.concat([self.train, self.test], ignore_index=True)
       
    def formatter(self):
        self.df = self.df[["tweet_text", "class_label"]]
        self.df = self.df.rename(columns={"tweet_text": "Text", "class_label": "Label"})
        self.test = self.test[["tweet_text", "class_label"]]
        self.test = self.test.rename(columns={"tweet_text": "Text", "class_label": "Label"})
        self.train = self.train[["tweet_text", "class_label"]]
        self.train = self.train.rename(columns={"tweet_text": "Text", "class_label": "Label"})
    
if __name__ == '__main__':
    cl = Checkthat2022Loader()
    print(cl.get_data())