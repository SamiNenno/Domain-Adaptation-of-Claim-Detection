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

class Checkthat2021Loader(DatasetLoader):
    def __init__(self):
        super().__init__()
        self.load_metadata("Checkthat2021")
        self.df = self.load_dataset()
        self.formatter()
        self.predefined_split = True
    
    def load_dataset(self):
        test_frames = []
        train_frames = []
        frames_2017 = []
        with ZipFile(self.dataset_path + 'v1.zip', 'r') as z:
            listOfiles = z.namelist()
            for tsv in listOfiles:
                if "v1/train/" in tsv and "tsv" in tsv:
                    with z.open(tsv) as f:
                        train_frames.append(pd.read_csv(f, header=None, delimiter="\t"))
                    if "2017" in tsv:
                        with z.open(tsv) as f:
                            frames_2017.append(pd.read_csv(f, header=None, delimiter="\t"))
                elif "v1/dev/" in tsv and "tsv" in tsv:
                    with z.open(tsv) as f:
                        test_frames.append(pd.read_csv(f, header=None, delimiter="\t"))
                    if "2017" in tsv:
                        with z.open(tsv) as f:
                            frames_2017.append(pd.read_csv(f, header=None, delimiter="\t"))
                else:
                    continue
        
        self.train = pd.concat(train_frames, ignore_index=True)
        self.test = pd.concat(test_frames, ignore_index=True)
        self._2017_df = pd.concat(frames_2017, ignore_index=True)
        return pd.concat([self.train, self.test], ignore_index=True)
       
    def formatter(self):
        self.df = self.df[[2, 3]]
        self.df = self.df.rename(columns={2: "Text", 3: "Label"})
        self.test = self.test[[2, 3]]
        self.test = self.test.rename(columns={2: "Text", 3: "Label"})
        self.train = self.train[[2, 3]]
        self.train = self.train.rename(columns={2: "Text", 3: "Label"})
        self._2017_df = self._2017_df[[2, 3]]
        self._2017_df = self._2017_df.rename(columns={2: "Text", 3: "Label"})
        
    def get_2017_data(self):
        return self._2017_df

    
if __name__ == '__main__':
    cl = Checkthat2021Loader()
    print(cl.get_2017_data()['Label'].value_counts())
    
