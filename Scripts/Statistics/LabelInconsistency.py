import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from Dataloader.DataCaller import DataCaller
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import plotly.express as px


class LabelInconsistencies(DataCaller):
    def __init__(self):
        super().__init__()
        self.data_names = ["claimbuster", "checkthat2019", "checkthat2021", "claimrank"]
        self.select(self.data_names)
        self.frame_dict = {}
        for name in self.data_names:
            self.loader[name].relabel('checkworthy')
            self.frame_dict[name] = self.loader[name].get_data()
        
    def extract_overlap(self, df1, df2):
        df = df1.copy() if df1.shape[0] <= df2.shape[0] else df2.copy()
        df = df.drop_duplicates(subset=['Text'])
        df_list = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc='Build Intersection', position=2, leave=False):
            sentence = row['Text']
            temp_df1 = df1[df1['Text'] == sentence].drop_duplicates(subset=['Text'])
            temp_df2 = df2[df2['Text'] == sentence].drop_duplicates(subset=['Text'])
            if temp_df1.shape[0] > 0 and temp_df2.shape[0] > 0:
                df_list.append(
                    pd.DataFrame.from_dict(
                        {
                           'Text' : [sentence],
                           'x' : temp_df1['Label'].to_list(),
                           'y' : temp_df2['Label'].to_list()
                        }
                    )
                )
        return pd.concat(df_list)
    
    def compute_accuracy(self):
        self.acc_matrix = np.zeros((len(self.data_names), len(self.data_names)))
        for x_coord, frame1 in enumerate(tqdm(self.data_names, total=len(self.data_names), desc='Compute Accuracy', position=0)):
            for y_coord, frame2 in enumerate(tqdm(self.data_names, total=len(self.data_names), desc=f"{frame1}", position=1, leave=False)):
                if self.acc_matrix[x_coord, y_coord] == 0.0:
                    if frame1 == frame2:
                        self.acc_matrix[x_coord, y_coord] = 1.0
                        self.acc_matrix[y_coord, x_coord] = 1.0
                    else:
                        df = self.extract_overlap(self.frame_dict[frame1], self.frame_dict[frame2])
                        acc = np.round(accuracy_score(y_true=df['x'].to_list(), y_pred=df['y'].to_list()), 2)
                        self.acc_matrix[x_coord, y_coord] = acc
                        self.acc_matrix[y_coord, x_coord] = acc
                    self.build_frame()        
                else:
                    continue
            self.save_frame()
        
    def build_frame(self):
        self.acc_frame = pd.DataFrame(data=self.acc_matrix, columns=self.data_names, index=self.data_names)
    def save_frame(self):
        self.acc_frame.to_csv(self.statistics_path + 'labelinconsistencies.csv', index=True)
    def plot(self):
        df = pd.read_csv(self.statistics_path + 'labelinconsistencies.csv')
        df = df.set_index('Unnamed: 0')
        fig = px.imshow(df, text_auto=True, color_continuous_scale='Aggrnyl')
        fig.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title=None, title='Label Consistency', xaxis={'side': 'top'})
        return fig
    def save_pic(self):
        fig = self.plot()
        fig.write_image(self.visuals_path + "PNG/LabelConsistency.png")
        fig.write_html(self.visuals_path + "HTML/LabelConsistency.html")
if __name__ == '__main__':
    lI = LabelInconsistencies()
    lI.save_pic()
            
        