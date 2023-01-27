from datetime import date
import pandas as pd
import numpy as np
from tqdm import tqdm
from codecarbon import EmissionsTracker
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from Utils.Ressources import Ressources
from Utils.Augmentation import DataAugmentation
from Dataloader.DataCaller import DataCaller



class DataSetup(DataCaller):
    def __init__(self):
        super().__init__()
        self.tracker = EmissionsTracker(project_name="DataSetup", output_dir=self.emission_path)
        self.tracker.start()
        self.select(["all"])
        self.random_seed = 2022
    
    def choose_one(self, dataset:str, split:bool = False, relabel:str="Original", augmentation:bool=True):
        if dataset in self.loader.keys() or dataset == "US-Debate2017":
            if dataset == "US-Debate2017":
                self.tracker.stop()
                return self.loader["checkthat2021"].get_2017_data()
            self.loader[dataset].relabel(relabel)
            if not split:
                df = self.loader[dataset].df
                if (dataset.lower() == 'checkthat2019' or dataset.lower() == 'checkthat2021') and augmentation:
                    DA = DataAugmentation(df, dataset.lower() + '_noSplit')
                    df = DA.return_data()
                self.tracker.stop()
                return df
            else:
                if self.loader[dataset].predefined_split:
                    train, test = self.loader[dataset].train, self.loader[dataset].test
                    if (dataset.lower() == 'checkthat2019' or dataset.lower() == 'checkthat2021') and augmentation:
                        DA = DataAugmentation(train, dataset.lower() + '_Split')
                        train = DA.return_data()
                    self.tracker.stop()
                    return train, test
                else:
                    train = self.loader[dataset].df.groupby("Label").sample(frac=0.7, random_state=self.random_seed).reset_index(drop=True)
                    test = self.loader[dataset].df.drop(train.index).reset_index(drop=True)
                    if (dataset.lower() == 'checkthat2019' or dataset.lower() == 'checkthat2021') and augmentation:
                        DA = DataAugmentation(train, dataset.lower() + '_Split')
                        train = DA.return_data()
                    self.tracker.stop()
                    return train, test
        else:
            self.tracker.stop()
            raise ValueError(f"This dataset does not exist. Choose one of the following:\n{list(self.loader.keys())+['US-Debate2017']}")
        
    def train_on_A_test_on_B(self, A:str, B:str, relabel:str="Original", augmentation:bool=True):
        train = self.choose_one(dataset=A, split=False, relabel=relabel, augmentation = augmentation)
        test = self.choose_one(dataset=B, split=False, relabel=relabel, augmentation = augmentation)
        train, test = self.drop_overlap(train, test)
        return train, test
    
    def drop_overlap(self, train, test):
        train["split"] = "Train"
        test["split"] = "Test"
        full = pd.concat([train, test], ignore_index=True)
        train = train[["Text", "Label"]]
        full = full.drop_duplicates(subset=["Text"])
        test = full[full["split"] == "Test"]
        test = test[["Text", "Label"]]
        return train.reset_index(drop=True),test.reset_index(drop=True)
    
    
if __name__ == '__main__':
    trainingsetup = DataSetup()
    df, test = trainingsetup.choose_one(dataset = 'claimrank', split = True, relabel="checkworthy")
    print(df["Label"].value_counts(), test["Label"].value_counts())
    