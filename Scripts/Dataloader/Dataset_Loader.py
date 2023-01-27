from codecs import ignore_errors

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from Utils.Ressources import Ressources


class DatasetLoader(Ressources):
    def __init__(self):
        self.load_paths()
        self.load_json()
        self.random_seed = 2022
        
    def load_metadata(self, dataset:str):
        self.metadata = self.metadata_dct[dataset]
        self.lab_to_idx = self.metadata['Labels']
        self.idx_to_lab = {value:key for key, value in self.lab_to_idx.items()}
        self.dataset_size = self.metadata['Size']
        self.num_labels = self.metadata['Num_labels']
        self.lang = self.metadata['Language']
        self.text_type = self.metadata['Text_Type']
        self.task_type = self.metadata['Task_Type']
        self.turn_factual = self.metadata['Turn_Factual']
        self.turn_checkworthy = self.metadata['Turn_Checkworthy'] 
        self.name = self.metadata['Name']
    
    def relabel(self, format:str="original"):
        if format.lower() == "factual":
            self.df['Label'] = self.df['Label'].apply(lambda x: self.turn_factual[str(x)])
        elif format.lower() == "checkworthy":
            self.df['Label'] = self.df['Label'].apply(lambda x: self.turn_checkworthy[str(x)])
        else:
            pass
        
    def get_name(self):
        return self.name
    def get_language(self):
        return self.lang
    def get_text_type(self):
        return self.text_type
    def get_task_type(self):
        return self.task_type
    def get_num_labels(self):
        return self.num_labels
    def get_labels(self):
        return [label for label in self.lab_to_idx.keys()]
    def get_size(self):
        return self.dataset_size
    def get_label_distribution(self, normalize:bool=False):
        return self.df['Label'].value_counts(normalize=normalize)
    def get_data(self, shuffle:bool = False):
        if shuffle:
            return self.df.sample(frac=1, random_state=self.random_seed, ignore_index=True)
        else:
            return self.df
    def get_predefined_split(self):
        if self.predefined_split:
            return self.train, self.test
        else:
            print("There is no predefined split!")
            return self.df, None
    
    
if __name__ == '__main__':
    dl = DatasetLoader()
    print("BLA")
