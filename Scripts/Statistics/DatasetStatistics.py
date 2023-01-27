import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from textblob_de import TextBlobDE
from textblob import TextBlob
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from Dataloader.DataCaller import DataCaller

class DatasetStatistics(DataCaller): 
    def __init__(self):
        super().__init__()
        self.US2017 = "US-Debate2017"
        self.select(["all"])
        
    def count_words(self):
        dct = {"Dataset":[], "Count":[]}
        for data_object in tqdm(self.get_all_data(), desc="Count words per sentence"):
            if len(data_object.get_language()) == 1 and data_object.get_language()[0] == "EN":
                counts = data_object.get_data()['Text'].apply(lambda x: len(TextBlob(x).words if type(x) is str else 0)).to_list()
            elif len(data_object.get_language()) == 1 and data_object.get_language()[0] == "DE":
                counts = data_object.get_data()['Text'].apply(lambda x: len(TextBlob(x).words if type(x) is str else 0)).to_list()
            else:
                continue
            dct["Count"] += counts
            dct["Dataset"] += [data_object.get_name()] * len(counts)
        #counts = self.loader["checkthat2021"].get_2017_data()['Text'].apply(lambda x: len(TextBlob(x).words)).to_list()
        #dct["Count"] += counts
        #dct["Dataset"] += [self.US2017] * len(counts)
        self.word_count = pd.DataFrame.from_dict(dct)
        self.word_count.to_csv(self.statistics_path + 'word_count.csv', index = False)
        
    def plot_word_distribution(self):
        word_count = pd.read_csv(self.statistics_path + 'word_count.csv')
        fig = px.box(word_count, 
                     y="Dataset", 
                     x="Count", 
                     template="plotly_white",
                     title="Word count per Sentence")
        fig.update_traces(marker_color='steelblue')
        fig.write_image(self.visuals_path + "PNG/WordDistribution.png", scale=7)
        fig.write_html(self.visuals_path + "HTML/WordDistribution.html")
        
    def plot_dataset_size(self):
        dct = {"Dataset":[], "Amount_Of_Sentences":[]}
        for data_object in tqdm(self.get_all_data(), desc="Plot amount of Sentences of Datasets"):
            dct["Dataset"].append(data_object.get_name())
            dct["Amount_Of_Sentences"].append(data_object.get_size())
        #dct["Dataset"].append(self.US2017)
        #dct["Amount_Of_Sentences"].append(self.loader["checkthat2021"].get_2017_data().shape[0])
        self.num_of_sentences = pd.DataFrame.from_dict(dct)
        self.num_of_sentences.to_csv(self.statistics_path + 'AmountSentences.csv', index = False)
        fig = px.bar(self.num_of_sentences,
                    y='Dataset', 
                    x='Amount_Of_Sentences',
                    template="plotly_white",
                    title="Amount of Sentences per Dataset")
        fig.update_traces(marker_color='steelblue')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        fig.update_layout({'xaxis_title':'Amount of Sentences'})
        fig.write_image(self.visuals_path + "PNG/AmountSentences.png", scale=7)
        fig.write_html(self.visuals_path + "HTML/AmountSentences.html")
    
    def save_label_distribution(self):
        self.distribution_list = []
        for data_object in tqdm(self.get_multi_label_data(), desc="Compute label distribution of Datasets"):
            df = data_object.get_label_distribution().to_frame().reset_index()
            df['label_name'] = df['index'].apply(lambda x: data_object.idx_to_lab[x])
            df['dataset'] = data_object.get_name()
            df.to_csv(self.statistics_path + data_object.get_name() + '_label_distribution.csv', index = False)
            self.distribution_list.append(df)
        #df = self.loader["checkthat2021"].get_2017_data()['Label'].value_counts().to_frame().reset_index()
        #df['label_name'] = df['index'].apply(lambda x: self.loader["checkthat2021"].idx_to_lab[x])
        #df['dataset'] = self.US2017
        #df.to_csv(self.statistics_path + self.US2017 + '_label_distribution.csv', index = False)
        #self.distribution_list.append(df)
      
    def plot_label_distribution(self):
        rows = 2
        cols = math.ceil(len(self.distribution_list) / rows)
        fig = make_subplots(rows=rows, 
                            cols=cols, 
                            specs=[[{"type": "pie"}]*cols ]*rows,
                            subplot_titles=[frame['dataset'][0] for frame in self.distribution_list])
        row = 1
        col = 1
        for frame in self.distribution_list:
            fig.add_trace(go.Pie(
            values=frame['Label'],
            labels=frame['label_name'],
            textinfo='label+percent',
            rotation=225,
            marker_colors=px.colors.qualitative.Pastel,
            marker=dict(line=dict(color='#000000', width=2)),
            name=frame['dataset'][0]), 
            row=row, col=col)
            col += 1
            if col > cols:
                col = 1
                row +=1
        fig.update_layout(title_text="Labeldistribution of Datasets")
        fig.update_layout(showlegend=False)
        fig.write_image(self.visuals_path + "PNG/LabelDistribution.png", scale=7)
        fig.write_html(self.visuals_path + "HTML/LabelDistribution.html")
        
    def bar_labeldistribution(self):
        temp = []
        for frame in self.distribution_list:
            if frame.shape[0] == 3:
                cw = frame[frame['index'] == 1]['Label'].sum()
                ncw = frame[frame['index'] != 1]['Label'].sum()
                frame = pd.DataFrame({
                    "index" : [0,1],
                    "Label" : [ncw, cw],
                    "label_name" : ['Non-check-worthy', 'Check-worthy'],
                    "dataset" : ['Claimbuster', 'Claimbuster']
                })
            temp.append(frame)
        distr_frame = pd.concat(temp)
        fig = px.bar(distr_frame, 
                x="Label", 
                y="dataset", 
                color="label_name", 
                color_discrete_sequence= px.colors.qualitative.Pastel, 
                template="plotly_white",
                orientation='h',title="Label Distribution")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        fig.update_layout({'xaxis_title':'Amount of Sentences'})
        fig.update_layout({'yaxis_title':''})
        fig.update_layout(legend=dict(title="Label"))
        fig.write_image(self.visuals_path + "PNG/BarDistribution.png", scale=7)
        fig.write_html(self.visuals_path + "HTML/BarDistribution.html")
        
    def plot_test_blend(self):
        path = '/home/sami/Claimspotting/Datasets/Blended_Testset.csv' ##! REPLACE BY FUNCTION
        testset = pd.read_csv(path)
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
        dct = {1:'checkworthy', 0:'not checkworthy'}
        df_label = testset.groupby(['cluster','Label']).size().reset_index().rename(columns={0:'count'})
        df_label['Label'] = df_label['Label'].apply(lambda x:dct[x])
        df_datasets = testset.groupby(['cluster','dataset']).size().reset_index().rename(columns={0:'count'})
        fig1 = px.bar(df_datasets, 
                x="cluster", 
                y="count", 
                color="dataset", 
                color_discrete_sequence= color_scheme, 
                template="plotly_white",
                orientation='v')
        fig1.update_layout({'xaxis_title':''})
        fig1.update_layout({'yaxis_title':'Amount of Sentences'})
        fig1.update_layout(legend=dict(title="Dataset"))
        fig2 = px.bar(df_label, 
                x="cluster", 
                y="count", 
                color="Label", 
                color_discrete_sequence= color_scheme, 
                template="plotly_white",
                orientation='v')
        fig2.update_layout(legend=dict(title="Label"))
        fig2.update_layout({'xaxis_title':'Cluster'})
        fig2.update_layout({'yaxis_title':'Amount of Sentences'})
        fig1.write_image(self.visuals_path + "PNG/BlendedTest_Datasets.png", scale=7)
        fig1.write_html(self.visuals_path + "HTML/BlendedTest_Datasets.html")
        fig2.write_image(self.visuals_path + "PNG/BlendedTest_Labels.png", scale=7)
        fig2.write_html(self.visuals_path + "HTML/BlendedTest_Labels.html")
        

            
    def fit(self):
        self.count_words()
        self.plot_word_distribution()
        self.plot_dataset_size()
        self.save_label_distribution()
        self.plot_label_distribution()
        self.bar_labeldistribution()
        self.plot_test_blend()
        

if __name__ == '__main__':
    ds = DatasetStatistics()
    ds.fit()
    