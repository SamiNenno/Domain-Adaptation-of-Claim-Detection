from LoadAllData import Loader
from Kmeans import Kmeans
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import plotly.express as px


class TrainTest():
    def __init__(self, split:float=.7, new_clusters:bool = True, start:int = 5, end:int = 100, by:int=5):
        self.split = split
        self.loader = Loader()
        self.model_path, self.result_path, self.vis_path, self.data_path, self.emission_path = self.loader.mkdir()
        self.k_range = list(range(start, end+1, by))
        if new_clusters:
            self.build_clusters()
        self.data = self.load_data()
        
    def build_clusters(self):
        self.loader.load_frames()
        df = self.loader.get_data()
        #df = df.sample(n=40000, random_state=2022) #! REMOVE
        kmeans = Kmeans(df, self.k_range)
        kmeans.fit(self.result_path, self.data_path, self.vis_path)
        
    def load_data(self):
        silhouette_scores = pd.read_csv(self.result_path + 'Silhouette_Results.csv')
        best_k = silhouette_scores.iloc[silhouette_scores['silh_score'].idxmax(),0]
        cluster_frame = pd.read_csv(self.data_path + 'ClusterFrame.csv')
        data = cluster_frame[['Text','Label', 'dataset']]
        data['cluster'] = cluster_frame[f"k={best_k}"]
        return data
    
    def train_test_split(self):
        self.data['DATASET_ID'] = self.data.index
        train_frame = self.data.groupby('cluster').sample(frac=self.split, random_state=2022)
        test_frame = self.data.drop(train_frame.index)
        return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)
    
class TripletTransformer(TrainTest):
    def __init__(self, 
                 split:float=.7, 
                 new_clusters:bool = True,
                 start:int = 5, 
                 end:int = 100, 
                 by:int = 5, 
                 epochs:int = 1, 
                 batch_size:int = 32, 
                 augmentation_factor:int = 20,
                 frac:float = .03):
        super().__init__(split, new_clusters, start, end, by)
        self.epochs = epochs
        self.batch_size = batch_size
        self.frac = frac
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1', device = 'cuda')
        self.train_frame, self.test_frame = self.train_test_split()
        self.test_frame.to_csv('/home/sami/Claimspotting/Datasets/Blended_Testset.csv', index = False)
        self.augmentation_factor = augmentation_factor
        #self.train_frame = self.train_frame.groupby('Label').sample(n=200) ##! REMOVE
        
    def build_dataset(self, df):
        def combine(df, augmentation_factor):
            df_list = []
            for label in range(2):
                for combination in range(int((augmentation_factor/2)+1)):
                    dct = dict(
                    anchor =  df[df['Label'] == label].sample(frac=1, random_state=combination)['Text'].to_list(),
                    positive =  df[df['Label'] == label].sample(frac=1, random_state=combination+augmentation_factor)['Text'].to_list(),
                    negative = df[df['Label'] != label].sample(frac=1, random_state=combination)['Text'].to_list()
                    )
                    temp = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dct.items() ])).dropna()
                    df_list.append(temp.query('anchor != positive'))
            return pd.concat(df_list, ignore_index=True).drop_duplicates().reset_index(drop=True)
        combination = combine(df, self.augmentation_factor)
        train_examples = []
        for row in tqdm(combination.itertuples(index=False), desc='Build Dataset', total=combination.shape[0]):
            train_examples.append(InputExample(texts=[row.anchor, row.positive, row.negative]))
        train_dataset = SentencesDataset(train_examples, self.model)
        return DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
    
    def train(self):
        path = '/home/sami/Claimspotting/Models/SentenceTransformer/'  #! Replace by function
        try:
            self.model = SentenceTransformer(path)
            print('\n--------------')
            print('Model already trained!')
            print('--------------\n')
        except Exception:
            train_loss = losses.TripletLoss(model=self.model)
            trainset = self.build_dataset(self.train_frame)
            self.model.fit(train_objectives=[(trainset, train_loss)], 
                        epochs=self.epochs, 
                        show_progress_bar=True,
                        save_best_model=True,
                        output_path= path
                        )
        
    def encode(self):
        train_X = self.model.encode(self.train_frame['Text'].to_list())
        test_X = self.model.encode(self.test_frame['Text'].to_list())
        train_y = self.train_frame['Label'].values
        test_y = self.test_frame['Label'].values
        return train_X, test_X, train_y, test_y
    
    def save_new_embeddings(self, train_X, test_X, train_y, test_y):
        df_train = pd.DataFrame(train_X)
        df_train['label'] = train_y
        df_test = pd.DataFrame(test_X)
        df_test['label'] = test_y
        df_train['DATASET_ID'] = self.train_frame['DATASET_ID']
        df_test['DATASET_ID'] = self.test_frame['DATASET_ID']
        df = pd.concat([df_train, df_test])
        df = df.sort_values(by=["DATASET_ID"]).reset_index(drop=True)
        df['dataset'] = self.data['dataset']
        df = df.drop(['DATASET_ID'], axis=1)
        df.to_csv('/home/sami/Claimspotting/Metadata/Results/SentenceTransformer/NewEmbeddings.csv', index = False)
    
class KNN():
    def __init__(self, 
                 max_k:int = 200,
                 split:float=.7, 
                 new_clusters:bool = True,
                 start:int = 5, 
                 end:int = 100, 
                 by:int = 5, 
                 epochs:int = 1, 
                 batch_size:int = 32, 
                 augmentation_factor:int = 20,
                 frac:float = .03):
        self.tt = TripletTransformer(split=split, 
                                     new_clusters = new_clusters, 
                                     start=start, 
                                     end=end, 
                                     by=by,
                                     augmentation_factor=augmentation_factor, 
                                     epochs=epochs)
        self.max_k = max_k
        
    def get_embeddings(self):
        self.tt.train()
        train_X, test_X, train_y, test_y = self.tt.encode()
        self.tt.save_new_embeddings(train_X, test_X, train_y, test_y)
        return train_X, test_X, train_y, test_y
        
    def predict(self, X_train, X_test, y_train, y_test):
        k_neighbors = list(range(5, self.max_k+1, 5))
        dct = {'ground_truth':list(y_test)}
        for k in tqdm(k_neighbors, total=len(k_neighbors), desc='KNN'):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            dct[f'k={k}'] = pred
            self.save(pd.DataFrame.from_dict(dct))
        return pd.DataFrame(dct)
        
    def plot(self):
        color_scheme = ['rgb(246, 207, 113)',
                    'rgb(248, 156, 116)',
                    'rgb(220, 176, 242)',
                    'rgb(135, 197, 95)',
                    'rgb(158, 185, 243)',
                    'rgb(254, 136, 177)',
                    'rgb(201, 219, 116)',
                    'rgb(139, 224, 164)',
                    'rgb(180, 151, 231)',
                    'rgb(102, 197, 204)',
                    'rgb(179, 179, 179)']
        ground_truth = self.results['ground_truth'].values
        dct_list = []
        for k_is in self.results.columns[1:]:
            pred = self.results[k_is].values
            dct_list.append(dict(
                K = int(k_is[2:]),
                Accuracy = accuracy_score(ground_truth, pred),
                F1 = f1_score(ground_truth, pred),
                Precision = precision_score(ground_truth, pred),
                Recall = recall_score(ground_truth, pred),
            ))
        df = pd.DataFrame(dct_list)
        max_f1 = df['K'].values[df['F1'].argmax()]
        df = df.melt(id_vars=['K'], value_vars=['Accuracy', 'F1', 'Precision', 'Recall'], var_name='Metric', value_name='Score')
        fig = px.line(df, x="K", y="Score", color='Metric',template='plotly_white', title='Metrics for K-Neighbors', color_discrete_sequence=color_scheme)
        fig.add_vline(x=max_f1, line_width=2, line_dash="dash", line_color='black')
        path = '/home/sami/Claimspotting/Metadata/Visuals/Clusters/' #! GET WITH FUNCTION
        fig.write_image(path + "KNN_Scores.png", scale=7)
        fig.write_html(path + "KNN_Scores.html")
    
    def fit(self):
        X_train, X_test, y_train, y_test = self.get_embeddings()
        self.results = self.predict(X_train, X_test, y_train, y_test)
        self.save(self.results)
        self.plot()
        
    def save(self, df):
        df.to_csv('/home/sami/Claimspotting/Metadata/Results/SentenceTransformer/KNN_Predictions.csv', index=False) #! GET WITH FUNCTION
        

if __name__ == '__main__':
    start = 5
    end = 110
    by = 5
    new_clusters = False
    split = .7
    max_k = 500
    epochs = 8
    augmentation_factor = 50 # 50 -> 1,623,743 sentences
    knn = KNN(max_k=max_k, split=split, new_clusters = new_clusters, start=start, end=end, by=by, augmentation_factor=augmentation_factor, epochs=epochs)
    #knn.get_embeddings()
    
    