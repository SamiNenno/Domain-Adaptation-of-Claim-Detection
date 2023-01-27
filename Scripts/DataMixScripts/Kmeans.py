import torch
import numpy as np
from kmeans_pytorch import kmeans
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
torch.manual_seed(2022)

class Kmeans():
    def __init__(self, df:pd.DataFrame, k_range:list):
        self.df = df
        self.k_range = k_range
        self.X = torch.tensor(self.df.iloc[:,2:-1].to_numpy())
        self.results = []
        
    def compute_cluster(self, k:int):
        cluster_labels, cluster_centers = kmeans(X=self.X, 
                                                 num_clusters=k, 
                                                 distance='cosine', 
                                                 iter_limit=100, 
                                                 seed=2022, 
                                                 device=torch.device('cuda:0'))
        return cluster_labels.cpu(), cluster_centers.cpu()
    
    def dimension_reduction(self, data_path):
        try:
            self.projection = pd.read_csv(data_path + 'TSNE_Frame.csv')
        except:
            self.pca_X = PCA(n_components=50).fit_transform(self.X)
            self.tsne_X = TSNE(n_components=2,random_state=2022, n_jobs=4, metric = 'cosine',learning_rate='auto', init='random').fit_transform(self.pca_X)
            self.projection = pd.DataFrame.from_dict(
                {
                    'X' : self.tsne_X[:,0],
                    'Y' : self.tsne_X[:,1],
                })
            self.projection.to_csv(data_path + 'TSNE_Frame.csv', index = False)
            
    def plot_cluster(self, cluster_labels):
        temp = self.projection.copy()
        temp["Cluster"] = cluster_labels
        fig = px.scatter(temp, x="X", y="Y", color="Cluster", opacity=.1, template='simple_white', color_continuous_scale=px.colors.sequential.RdBu)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig
    
    def plot_silhouette(self, path):
        df = pd.DataFrame(self.results)
        max_cluster = df['clusters'].values[df['silh_score'].argmax()]
        fig = px.line(df, x='clusters', y="silh_score", template="plotly_white", title="Silhouette Score for k-Clusters")
        fig.update_layout(xaxis_title="k-Clusters", yaxis_title="Silhouette Score")
        fig.add_vline(x=max_cluster, line_width=2, line_dash="dash", line_color='black')
        fig.write_image(path + 'Silhouette.png', scale = 7)
        
    def fit(self, result_path, data_path, vis_path):
        print('Dimension Reduction...')
        self.dimension_reduction(data_path)
        print('Done!')
        for idx, k in enumerate(self.k_range):
            cluster_labels, cluster_centers = self.compute_cluster(k=k)
            score = silhouette_score(self.X, cluster_labels, metric='cosine')
            self.results.append(dict(
                clusters = k,
                silh_score = score))
            self.df[f"k={k}"] = cluster_labels
            self.save_results(result_path, data_path)
            if idx > 0:
                self.plot_silhouette(vis_path)
            fig = self.plot_cluster(cluster_labels)
            self.save_fig(fig, vis_path, k)
            
    def save_results(self, result_path, data_path):
        pd.DataFrame(self.results).to_csv(result_path + 'Silhouette_Results.csv', index=False)
        self.df.to_csv(data_path + 'ClusterFrame.csv', index = False)
        
    def save_fig(self, fig, path, k):
        fig.write_image(f"{path}{k}_Cluster.png", scale=7)
        fig.write_html(f"{path}{k}_Cluster.html")
    
