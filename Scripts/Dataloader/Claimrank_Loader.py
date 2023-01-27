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

class ClaimrankLoader(DatasetLoader):
    def __init__(self):
        super().__init__()
        self.load_metadata("CW-USPD-2016")
        self.df = self.load_dataset()
        self.formatter()
        self.predefined_split = False
    
    def load_dataset(self):
        df_list = []
        with ZipFile(self.dataset_path + 'apepa_claim-rank_master_data.zip', 'r') as z:
            listOfiles = z.namelist()
            for tsv in listOfiles:
                if "transcripts_all_sources/" in tsv and "tsv" in tsv:
                    with z.open(tsv) as f:
                        frame = pd.read_csv(f, delimiter="\t")
                        df_list.append(frame)
                else:
                    continue
        
        self.df = pd.concat(df_list, ignore_index=True)
        return self.df
       
    def formatter(self):
        self.df['Label'] = self.df['ALL'].apply(lambda x: 1 if x > 0 else 0)
        self.df = self.df[['Text', 'Label']]
        
 
    
if __name__ == '__main__':
    cl = ClaimrankLoader()
    print(cl.df)
    
