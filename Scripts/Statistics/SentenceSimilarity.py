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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px

class VisualEmbeddings(Ressources):
    def __init__(self) -> None:
        self.load_paths()
        self.load_json()
        self.path = self.embeddings_path + 'not_augmented/*.csv'
        self.color_scheme = ['rgb(246, 207, 113)',
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
        
    def load_frames(self):
        def helper(x):
            if int(x) == 1:
                return 'Checkworthy'
            else:
                return 'Not-Checkworthy'
        df_list = []
        files = glob.glob(self.path)
        for csv in tqdm(files, total=len(files), desc='Load CSV-Files', leave=False):
            dataset = csv.split('/')[-1].split('_')[0]
            df = pd.read_csv(csv)
            df['dataset'] = dataset
            df_list.append(df)
        self.df = pd.concat(df_list)
        self.df.dropna(inplace=True)
        self.df['Label'] = self.df['Label'].apply(lambda x: helper(x))
        self.df = self.df.sample(frac=1, random_state=2022)
        self.df = self.paint_frame(self.df)
        #self.df = self.df.head(3000) #! REMOVE
    
    def get_color_dict(self, label_list):
        return {label:self.color_scheme[idx] for idx, label in enumerate(label_list)}
    
    def color_style(self, label, dct):
        return f'<span style="color: {dct[label]}">{label}</span>'
        
    def paint_frame(self,df):
        label_dct = self.get_color_dict(df['Label'].unique().tolist())
        dataset_dct = self.get_color_dict(df['dataset'].unique().tolist())
        df['Label'] = df['Label'].apply(lambda x: self.color_style(x, label_dct))
        df['dataset'] = df['dataset'].apply(lambda x: self.color_style(x, dataset_dct))
        return df
    
    def build_dataset(self):
        self.y_label = self.df['Label'].to_numpy().reshape(-1)
        self.y_dataset = self.df['dataset'].to_numpy().reshape(-1)
        self.X = self.df.iloc[:,2:-1].to_numpy()
    
    def pca(self):
        self.pca_X = PCA(n_components=50).fit_transform(self.X)
        
    def tsne(self):
        self.tsne_X = TSNE(n_components=2,random_state=2022, n_jobs=4, metric = 'cosine',learning_rate='auto', init='random').fit_transform(self.pca_X) 
        
    def plot_and_save(self):
        fig = self.plot('Label')
        self.save(fig, 'Label')
        fig = self.plot('Dataset')
        self.save(fig, 'Dataset')
        
    def plot(self, color):
        fig = px.scatter(self.projection, x="X", y="Y", color=color, opacity=.03, template='simple_white', color_discrete_sequence=self.color_scheme )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig
    
    def save(self, fig, color):
        fig.write_image(self.visuals_path +f"PNG/Embedding_TSNE_{color}.png", scale=7)
        fig.write_html(self.visuals_path + f"HTML/Embedding_TSNE_{color}.html")
            
    def fit(self):
        pipeline = [
            self.load_frames,
            self.build_dataset,
            self.pca,
            self.tsne]
        for task in tqdm(pipeline, total=len(pipeline), desc='Pipeline', leave=False):
            task()
        self.projection = pd.DataFrame.from_dict(
            {
                'X' : self.tsne_X[:,0],
                'Y' : self.tsne_X[:,1],
                'Label' : self.y_label,
                'Dataset' : self.y_dataset
            }
        )
        self.plot_and_save()
        
    
            
if __name__ == '__main__':
    tsne = VisualEmbeddings()
    tsne.fit()
