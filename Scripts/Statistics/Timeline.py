import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from Utils.Ressources import Ressources
import pandas as pd
import plotly.express as px
import json
import pendulum

class Timeline(Ressources):
    def __init__(self):
        self.load_paths()
        self.load_json()
        self.PATH = self.metadata_path + 'Datasets.json'
        with open(self.PATH, 'r') as f:
            self.data = json.load(f)
        self.df = self.extract()
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
    
    def extract(self):
        temp = []
        for key in self.data.keys():
            dct = self.data[key]
            name = dct['Name']
            if name == 'MultiFC':
                continue
            start = dct['Period']['Start']
            end = dct['Period']['End']
            source = dct['Source'] 
            temp.append(dict(
                Dataset = name,
                Datasource = source,
                Start = pendulum.from_format(start, 'DD-MM-YYYY'),
                End = pendulum.from_format(end, 'DD-MM-YYYY')
            ))
        return pd.DataFrame(temp)
    
    def plot(self):
        fig = px.timeline(self.df, 
                          x_start="Start", 
                          x_end="End", 
                          y="Dataset",
                          color="Datasource",
                          template="plotly_white",
                          color_discrete_sequence=self.color_scheme,
                          title="Data set Timeline")
        fig.update_yaxes(autorange="reversed")
        fig.write_image(self.visuals_path + "PNG/Timeline.png", scale=7)
        fig.write_html(self.visuals_path + "HTML/Timeline.html")
if __name__ == '__main__':
    tl = Timeline().plot()