import os
import pandas as pd
import json
from datetime import datetime

class Ressources():
    def load_paths(self):
        self.project_path = os.getcwd().split("Claimspotting")[0] + "Claimspotting/"
        self.dataloader_path = self.project_path + "Dataloader/"
        self.dataset_path = self.project_path + "Datasets/"
        self.metadata_path = self.project_path + "Metadata/"
        self.model_path = self.project_path + "Models/"
        self.xgboost_path = self.model_path + 'XGBoost/'
        self.statistics_path = self.metadata_path + "Statistics/"
        self.visuals_path = self.metadata_path + "Visuals/"
        self.emission_path = self.metadata_path + "Emissions/"
        self.result_path = self.metadata_path + "Results/"
        self.fail_path = self.metadata_path + "FAILED_TRIALS/"
        self.prediction_path = self.metadata_path + "Predictions/"
        self.augmentation_path = self.dataset_path + 'AugmentedPositives/'
        self.embeddings_path = self.dataset_path + 'SentenceEmbeddings/'

        
    def load_json(self):
        with open(self.metadata_path + 'Datasets.json', 'r') as f:
            self.metadata_dct = json.load(f)
        
    def folder_format(self, path):
        return path if path[-1] == '/' else path + '/'
    
    def document_failure(self, dct):
        dct["time"] = str(datetime.now())
        f = self.fail_path + "Fails.csv"
        new_fail = pd.DataFrame.from_dict({key:[value] for key, value in dct.items()})
        if os.path.exists(f):
            fail_frame = pd.read_csv(f)
            fail_frame = pd.concat([fail_frame, new_fail], ignore_index=True)
        else:
            fail_frame = new_fail
        fail_frame.to_csv(f, index=False)
        
if __name__ == '__main__':
    ress = Ressources()
    ress.load_paths()
    ress.load_json()
    ress.document_failure({"hallo":"tschüss"})