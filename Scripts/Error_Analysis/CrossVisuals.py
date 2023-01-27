import plotly.figure_factory as ff
import pandas as pd
import numpy as np

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from Utils.Ressources import Ressources


class Heat(Ressources):
    def __init__(self):
        self.load_paths()
        self.load_json()
        self.df = pd.read_csv(self.result_path + 'Results.csv')

    def return_by_dataset(self, dataset):
        return self.df[self.df['Train'] == dataset].drop_duplicates(subset=['Model', 'Train', 'Test'])

    def format_model_names(self, df):
        def helper(model_name):
            if model_name.lower() not in ['logreg', 'svm']:
                return 'transformer'
            else:
                if model_name == 'logreg':
                    return 'logistic regression'
                else:
                    return 'svm'
        df['Model'] = df['Model'].apply(lambda x: helper(x))
        return df


    def format_for_heatmap(self,df):
        return df[['Model', "Train", "Test", "Accuracy", "F1", "Recall", "Precision"]].reset_index(drop=True)

    def add_display(self, df):
        to_be_displayed = []
        for idx,row in df.iterrows():
            if row['Test'] == "multifc":
                to_be_displayed.append(row['Recall'])
            else:
                to_be_displayed.append(row['F1'])
        df['to_be_displayed']=to_be_displayed
        return df

    def create_heatmap(self,df, dataset, metric):
        cols = []
        for model in df['Model'].unique():
            cols.append(df[df['Model'] == model][metric].to_numpy().reshape(-1,1))
        z = np.round(np.concatenate(cols, axis=1),2)


        x = list(self.format_model_names(df)['Model'].unique())
        y = list(df['Test'].unique())


        fig = ff.create_annotated_heatmap(
                                        z,
                                        x=x,
                                        y=y,
                                        annotation_text=z,
                                        colorscale='Tealgrn',
                                        )
        fig.update_layout(title=dict(text=f'Trained on {dataset} dataset (Metric: {metric})', font_size=18))
        fig.update_xaxes(tickfont_size=13)
        fig.update_yaxes(tickfont_size=13)
        return fig

    def plot(self, dataset, metric):
        result = self.return_by_dataset(dataset=dataset)
        result = self.format_for_heatmap(result)
        # Helper because dataset is incomplete
        result = result[result['Model'] != 'xlm-roberta-base'] #! remove
        result = result[(result['Test'] != 'germeval')] #! remove
        result = self.add_display(result)
        fig = self.create_heatmap(df=result, dataset=dataset, metric = metric)
        self.save(fig, dataset=dataset, metric=metric)

    def save(self, fig, dataset, metric):
        fig.write_image(self.visuals_path + f"PNG/{dataset}_{metric}.png")
        fig.write_html(self.visuals_path + f"HTML/{dataset}_{metric}.html")

    def plot_all(self):
        datasets = self.df['Train'].unique()
        for dataset in datasets:
            try:
                self.plot(dataset=dataset)
            except Exception as e:
                print(e)
                pass
if __name__ == '__main__':
    heat = Heat()
    heat.plot("claimbuster", "Recall")