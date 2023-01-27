import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from Utils.Ressources import Ressources
import glob
from tqdm import tqdm
import pandas as pd

class Loader(Ressources):
    def __init__(self) -> None:
        self.load_paths()
        self.load_json()
        self.path = self.embeddings_path + 'not_augmented/*.csv'
        
    def load_frames(self):
        df_list = []
        files = glob.glob(self.path)
        for csv in tqdm(files, total=len(files), desc='Load CSV-Files', leave=False):
            dataset = csv.split('/')[-1].split('_')[0]
            df = pd.read_csv(csv)
            df['dataset'] = dataset
            df_list.append(df)
        self.df = pd.concat(df_list)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True)
        self.df['Label'] = self.df['Label'].astype('int')
        
    def get_data(self):
        return self.df.sample(frac=1, random_state=2022).reset_index(drop=True)
        
    def mkdir(self):
        model_path = self.model_path + 'SentenceTransformer/'
        if not os.path.exists(model_path):
            os.makedirs(model_path) 
        result_path = self.result_path + 'SentenceTransformer/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        vis_path = self.visuals_path + 'Clusters/'
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        data_path = self.dataset_path + 'Clusters/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        return (
            model_path,
            result_path,
            vis_path,
            data_path,
            self.emission_path
        )
        
if __name__ == '__main__':
    loader = Loader()
    loader.load_frames()
    dirs = loader.mkdir()
    df = loader.get_data()
    
    print()
    