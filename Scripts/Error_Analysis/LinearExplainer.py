from sklearn.pipeline import make_pipeline
import joblib
import pickle
import json
from tqdm import tqdm
import numpy as np
from lime.lime_text import LimeTextExplainer
import os
import sys
from codecarbon import EmissionsTracker

## https://arxiv.org/pdf/1602.04938.pdf
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from Dataloader.TrainingSetup import DataSetup

class WeightDict():
    def __init__(self):
        self.positives = {}
        self.negatives = {}

    def add_value(self, dct, key, value):
        if key in dct:
            dct[key].append(value)
        else:
            dct[key] = [value]
        return dct

    def add_weight(self, label, key, value):
        if label == 0:
            self.negatives = self.add_value(self.negatives, key, value)
        else:
            self.positives = self.add_value(self.positives, key, value)

    def get_positives(self):
        return self.positives
    def get_negatives(self):
        return self.negatives

class LimeExplainer(WeightDict):
    def __init__(self, model_path, vectorizer_path):
        super().__init__()
        self.model = joblib.load(model_path)
        self.vectorizer = pickle.load(open(vectorizer_path,'rb'))
        self.pipeline = make_pipeline(self.vectorizer, self.model)
        self.class_names = ['non-checkworthy', 'checkworthy']
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def word_weight(self, sentence, label, num_features=30):
        self.exp = self.explainer.explain_instance(sentence, self.pipeline.predict_proba, num_features=num_features)
        self.insert_weight(self.exp.as_list(), label)

    def insert_weight(self, weight_list, label):
        for word, weight in weight_list:
            self.add_weight(label, word, weight)

class LinearExplainer(DataSetup):
    def __init__(self) -> None:
        super().__init__()
        self.load_paths()
        self.load_json()
        self.baseline_path = self.model_path + 'Baseline/'
        self.folders = [dir for dir in os.listdir(self.baseline_path) if 'Train_on' in dir]

    def iterator(self):
        for folder in tqdm(self.folders, desc='Baselines', total = len(self.folders), position=0):
            try:
                path = self.baseline_path + folder + '/'
                model, _, _, trainset, _, _, testset, task_type = tuple(folder.split('_'))
                model_path = path + 'model.joblib'
                vectorizer_path = path + 'vectorizer.pkl'
                explainer = LimeExplainer(model_path=model_path, vectorizer_path=vectorizer_path)
                df = self.choose_one(dataset=testset, split=False, relabel=task_type, augmentation = False)
                yield explainer, df, path
            
            except FileNotFoundError as e:
                print(e)

    def explain(self):
        def n_important(dct, n = 50):
            keys = []
            values = []
            for key, value in dct.items():
                keys.append(key)
                values.append(np.mean(np.absolute(np.array(value))))
            top_keys = np.array(keys)[np.argsort(values)[-n:]]
            return {key:dct[key] for key in top_keys}
        for explainer, df, path in self.iterator():
            for  idx, row in tqdm(df.iterrows(), desc = 'Explain', total=df.shape[0], position=1):
                explainer.word_weight(row['Text'], row['Label'])
            yield n_important(explainer.get_positives()), n_important(explainer.get_negatives()), path
            
    def get_explanations(self):
        tracker = EmissionsTracker(project_name="LinearExplainer", output_dir=self.emission_path)
        tracker.start()
        for positive, negative, path in self.explain():
            with open(path + 'positive_weights.json', 'w') as json_file:
                json.dump(positive, json_file, indent=6)
            with open(path + 'negative_weights.json', 'w') as json_file:
                json.dump(negative, json_file, indent=6)
        tracker.stop()  
 
 



if __name__ == '__main__':
    
    linex = LinearExplainer()
    linex.get_explanations()
    

