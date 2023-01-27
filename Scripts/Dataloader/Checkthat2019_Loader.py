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

class Checkthat2019Loader(DatasetLoader):
    def __init__(self):
        super().__init__()
        self.load_metadata("Checkthat2019")
        self.df = self.load_dataset()
        self.formatter()
        self.predefined_split = True
    
    def load_dataset(self):
        test_frames = []
        train_frames = []
        with ZipFile(self.dataset_path + 'apepa clef2019-factchecking-task1 master data.zip', 'r') as z:
            listOfiles = z.namelist()
            for tsv in listOfiles:
                if "training/" in tsv and "tsv" in tsv:
                    with z.open(tsv) as f:
                        train_frames.append(pd.read_csv(f, header=None, delimiter="\t"))
                elif "test_annotated/" in tsv and "tsv" in tsv:
                    with z.open(tsv) as f:
                        test_frames.append(pd.read_csv(f, header=None, delimiter="\t"))
                else:
                    continue
        
        self.train = pd.concat(train_frames, ignore_index=True)
        self.test = pd.concat(test_frames, ignore_index=True)
        return pd.concat([self.train, self.test], ignore_index=True)
       
    def formatter(self):
        self.df = self.df[[2, 3]]
        self.df = self.df.rename(columns={2: "Text", 3: "Label"})
        self.test = self.test[[2, 3]]
        self.test = self.test.rename(columns={2: "Text", 3: "Label"})
        self.train = self.train[[2, 3]]
        self.train = self.train.rename(columns={2: "Text", 3: "Label"})
        
 
    
if __name__ == '__main__':
    cl = Checkthat2019Loader()
    print(cl.get_data())
    print(cl.get_label_distribution())
    
