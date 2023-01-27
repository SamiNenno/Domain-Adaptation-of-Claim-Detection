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

class ClaimbusterLoader(DatasetLoader):
    def __init__(self):
        super().__init__()
        self.load_metadata("Claimbuster")
        self.df = self.load_dataset()
        self.formatter()
        self.predefined_split = False
    
    def load_dataset(self):
        with ZipFile(self.dataset_path + 'ClaimBuster_Datasets.zip', 'r') as z:
            with z.open('ClaimBuster_Datasets/datasets/crowdsourced.csv') as f:
                df1 = pd.read_csv(f)
            with z.open('ClaimBuster_Datasets/datasets/groundtruth.csv') as f:
                df2 = pd.read_csv(f)
        return pd.concat([df1, df2], ignore_index=True)
    
    def formatter(self):
        self.df = self.df[["Text", "Verdict"]]
        self.df = self.df.rename(columns={"Verdict": "Label"})

    
if __name__ == '__main__':
    cl = ClaimbusterLoader()
    cl.relabel("checkworthy")
    
    
