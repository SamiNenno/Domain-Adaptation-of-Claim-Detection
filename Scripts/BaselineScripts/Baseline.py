import random
import json
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm 
from codecarbon import EmissionsTracker
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn.model_selection import StratifiedKFold
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from HanTa import HanoverTagger as ht
#https://sklearn-genetic-opt.readthedocs.io/en/stable/tutorials/basic_usage.html
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from Dataloader.TrainingSetup import DataSetup
from Utils.Evaluator import Evaluator


class Baseline(DataSetup):
    def __init__(self, 
                 modelname:str, 
                 train_name:str, 
                 test_name:str="NONE", 
                 relabel:str="Original",
                 n_splits = 3): 
        super().__init__()
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        self.modelname = modelname.lower()
        self.relabel = relabel
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
        self.cpus = 10
        self.generations = 7 
        self.scoring_metric = 'f1' if 'checkthat' in self.modelname else 'accuracy'
        self.train_name, self.test_name = train_name, test_name
       
        if self.train_name == self.test_name:
            self.train, self.test = self.choose_one(dataset=self.train_name, split=True, relabel=relabel)
            self.test_frame = None
        else:
            self.train, self.test = self.train_on_A_test_on_B(A=self.train_name, B=self.test_name, relabel=relabel)
            self.test_frame = self.test.copy()
        self.name = f"{self.modelname}_Train_on_{self.train_name}_test_on_{self.test_name}_{relabel}"  
        #self.train, self.test = self.train.groupby("Label").sample(n=50, random_state=22, replace=True), self.test.sample(n=50, random_state=22, replace=True)            
        self.model_path = self.model_path + "Baseline/" + self.name + "/"
        self.language = self.loader[self.train_name].get_language()[0]
    
    def setup_folder(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
    
    def BoW(self):
        self.preprocessing()
        if os.path.exists(self.model_path + 'vectorizer.pkl'):
            self.vectorizer = joblib.load(self.model_path + "vectorizer.pkl")
        else:
            self.vectorizer = CountVectorizer(min_df=2)
            self.vectorizer.fit(pd.concat([self.train, self.test])['Text'].to_list())
            joblib.dump(self.vectorizer, self.model_path + "vectorizer.pkl")
            with open(self.model_path + 'BoW.json', 'w') as json_file:
                json.dump(self.vectorizer.vocabulary_, json_file, indent=6)
            
        self.X_train = self.vectorizer.transform(self.train['Text'])
        self.X_test = self.vectorizer.transform(self.test['Text'])
        self.y_train = np.array(self.train['Label'].to_list())
        self.y_test = np.array(self.test['Label'].to_list())
        
              
    def train_(self):
        if self.modelname == "logreg":
            param_grid = {'C': Integer(1,100), 
                        'penalty': Categorical(['l1', 'l2']), 
                        "class_weight":Categorical(['balanced']),
                        "solver":Categorical(["liblinear", "saga"]),
                        "max_iter":Integer(1000, 10000)}
            self.model = LogisticRegression(random_state=2022)
        else:
            self.model = SVC(probability=True)
            param_grid = {
                "C" : Continuous(1e0, 1e3, distribution='log-uniform'),
                "gamma" : Continuous(1e-4, 1e-3, distribution='log-uniform'),
                "class_weight":Categorical(['balanced', None]),
                "kernel":Categorical(["rbf"]),
            }
        self.evolved_estimator = GASearchCV(estimator=self.model,
                            cv=self.cv,
                            scoring=self.scoring_metric,
                            param_grid=param_grid,
                            n_jobs=self.cpus, 
                            generations = self.generations, 
                            population_size=4,
                            algorithm = 'eaMuPlusLambda',
                            verbose=True)
        self.evolved_estimator.fit(self.X_train, self.y_train)
        self.model = self.evolved_estimator.best_estimator_
        joblib.dump(self.model, self.model_path + 'model.joblib')
        self.best_params = self.evolved_estimator.best_params_
        self.history = self.evolved_estimator.history
        with open(self.model_path + 'BestParams.json', 'w') as json_file:
            json.dump(self.best_params, json_file, indent=6)
        pd.DataFrame.from_dict(self.history).to_csv(self.model_path + 'TrainingHistory.csv', index = False)
            
    def test_(self):
        self.probs = self.model.predict_proba(self.X_test)
        evaluator = Evaluator(model = self.modelname,
                train = self.train_name,
                test = self.test_name,
                test_frame = None,#self.test_frame,
                task_type = self.relabel,
                y_true = self.y_test, 
                y_probs = self.probs,
                )
        evaluator.evaluate()

    def preprocessing(self):
        def remove_stop_words(sentence, stops,language):
            words = word_tokenize(sentence, language=language)
            wordsFiltered = []
            for w in words:
                if w not in stops:
                    wordsFiltered.append(w)
            return " ".join(wordsFiltered)
        
        def lemmatize(sentence, language, lemmatizer):
            if language == "german":
                return " ".join([lemmatizer.analyze(word)[0] for word in word_tokenize(sentence, language=language)])
            else: 
                return " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(sentence, language=language)])
            
        lan = 'english' if self.language == "EN" else 'german'
        stops = set(stopwords.words(lan))
        lemmatizer = WordNetLemmatizer() if self.language == "EN" else ht.HanoverTagger('morphmodel_ger.pgz')
        self.train['Text'] = self.train['Text'].apply(lambda x: x.lower())
        self.train['Text'] = self.train['Text'].apply(lambda x: remove_stop_words(x, stops=stops, language=lan))
        self.train['Text'] = self.train['Text'].apply(lambda x : lemmatize(x, language=lan, lemmatizer=lemmatizer))
        self.test['Text'] = self.test['Text'].apply(lambda x: x.lower())
        self.test['Text'] = self.test['Text'].apply(lambda x: remove_stop_words(x, stops=stops, language=lan))
        self.test['Text'] = self.test['Text'].apply(lambda x : lemmatize(x, language=lan, lemmatizer=lemmatizer))
        
        
    def fit(self):
        self.tracker = EmissionsTracker(project_name=f"Baseline_{self.modelname}", output_dir=self.emission_path+'Baseline/')
        self.tracker.start()
        self.setup_folder()
        for step in tqdm(list(range(3)), desc='Vectorize, Train, Test'):
            if step == 0:
                self.BoW()
            if step == 1:
                self.train_()
            if step == 2:
                self.test_()
        self.tracker.stop()  
        
        
    
    
if __name__ == '__main__':
    ml = Baseline('SVM', 'claimrank', 'claimbuster', 'checkworthy', 2)
    ml.fit()
    
