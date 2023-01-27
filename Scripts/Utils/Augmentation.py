import nlpaug.augmenter.word as naw
import pandas as pd
from tqdm import tqdm
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from Utils.Ressources import Ressources

class DataAugmentation(Ressources):
    def __init__(self, df, dataset) -> None:
        self.load_paths()
        self.load_json()
        self.columns = df.columns.to_list()
        self.split(df)
        self.sentence_list = self.positives.iloc[:,0].to_list()
        self.augmentation_path += f"{dataset}.csv"
        self.dataset = dataset
        self.translation1 = naw.BackTranslationAug(from_model_name = 'Helsinki-NLP/opus-mt-en-de', to_model_name = 'Helsinki-NLP/opus-mt-de-en', device='cuda')
        self.translation2 = naw.BackTranslationAug(from_model_name = 'Helsinki-NLP/opus-mt-en-fr', to_model_name = 'Helsinki-NLP/opus-mt-fr-en', device='cuda')
        self.translation3 = naw.BackTranslationAug(from_model_name = 'Helsinki-NLP/opus-mt-en-es', to_model_name = 'Helsinki-NLP/opus-mt-es-en', device='cuda')
        self.translation4 = naw.BackTranslationAug(from_model_name = 'Helsinki-NLP/opus-mt-en-da', to_model_name = 'Helsinki-NLP/opus-mt-da-en', device='cuda')
        self.synonym1 = naw.SynonymAug(aug_p = 0.3)
        self.synonym2 = naw.SynonymAug(aug_p = 0.7)
        self.augmentation_functions = [
            self.translation1,
            self.translation2,
            #self.translation3,
            #self.translation4,
            self.synonym1,
            #self.synonym2
        ]
        self.seed = 2022

    def split(self, df):
        dataframes = [group for _, group in df.groupby(self.columns[1])]
        self.negatives = dataframes[0]
        self.positives = dataframes[1]


    def augment(self):
        for augmentation in tqdm(self.augmentation_functions, desc = 'Data Augmentation'):
            augmented_sentences = augmentation.augment(self.sentence_list)
            self.sentence_list += augmented_sentences
        self.df = pd.DataFrame.from_dict({self.columns[0] : self.sentence_list, self.columns[1] : 1})
        self.df = self.df.drop_duplicates().reset_index(drop=True)


    def return_data(self):
        try:
            self.df = pd.read_csv(self.augmentation_path)
            print('Augmented dataset already exists!')
            print(f'Original numner: {self.positives.shape[0]}')
            print(f'With Augmentation: {self.df.shape[0]}')
        except FileNotFoundError:
            self.augment()
            print(f'Original numner: {self.positives.shape[0]}')
            print(f'With Augmentation: {self.df.shape[0]}')
            self.df.to_csv(self.augmentation_path, index = False)
        self.df = pd.concat([self.df, self.negatives])
        self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        return self.df

    

if __name__ == '__main__':
    data = [
        "We will be a country of generosity and warmth",
        "But we will also be a country of law and order",
        "Our Convention occurs at a moment of crisis for our nation",
        "The attacks on our police, and the terrorism of our cities, threaten our very way of life.",
        "Any politician who does not grasp this danger is not fit to lead our country.",
        "Americans watching this address tonight have seen the recent images of violence in our streets and the chaos in our communities.",
    ]   
    df = pd.DataFrame({'text': data})
    df['label']= [1, 0, 1, 1, 0, 1]
    DA = DataAugmentation(df, "test")
    df2 = DA.return_data()
    print(df2['label'].value_counts())