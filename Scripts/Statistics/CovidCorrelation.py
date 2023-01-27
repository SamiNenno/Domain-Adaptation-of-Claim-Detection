from zipfile import ZipFile
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
import scipy.stats
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from Dataloader.Dataset_Loader import DatasetLoader



class CovidCorrelation(DatasetLoader):
    def __init__(self):
        super().__init__()
        self.rename_dict = {
        "q1_label" : "Factuality",
        "q2_label" : "False Information",
        "q3_label" : "Public Interest",
        "q4_label" : "Harmful Ordinal",
        "q5_label" : "Checkworthy",
        "q6_label" : "Harmful Categorical",
        "q7_label" : "Government"
        }
        
        self.df = self.load_dataset(binary = True)
        self.correlation_df = self.correlation(self.df)
        self.save(binary = True)
        self.df = self.load_dataset(binary = False)
        self.correlation_df = self.correlation(self.df)
        self.save(binary = False)

    def load_dataset(self, binary = True):
        scale = "binary" if binary else "multiclass"
        with ZipFile(self.dataset_path + 'firojalam COVID-19-disinformation master data-english.zip', 'r') as z:
            with z.open(f'covid19_disinfo_english_{scale}_train.tsv') as f:
                train = pd.read_csv(f, delimiter="\t")
            with z.open(f'covid19_disinfo_english_{scale}_dev.tsv') as f:
                dev = pd.read_csv(f, delimiter="\t")
            with z.open(f'covid19_disinfo_english_{scale}_test.tsv') as f:
                test = pd.read_csv(f, delimiter="\t")
        df = pd.concat([train, dev, test], ignore_index=True)
        df = df.dropna(subset=["q5_label"])
        df = df.rename(columns=self.rename_dict)
        df = df.drop(columns=["tweet_id"])
        df = df.reset_index(drop=True)
        df['False Public']=(df['False Information'] == 'yes') & (df['Public Interest'] == 'yes')
        df['False Public'] = df['False Public'].apply(lambda x: 'yes' if x else 'no')
        df['Harmful Public']=(df['Harmful Ordinal'] == 'yes') & (df['Public Interest'] == 'yes')
        df['Harmful Public'] = df['Harmful Public'].apply(lambda x: 'yes' if x else 'no')
        return df
        
    def matthew_correlation(self, df):
        dct = {}
        for col in df.columns:
            dct[col] = [matthews_corrcoef(df['Checkworthy'].to_list(), df[col].to_list())]
        return pd.DataFrame.from_dict(dct)
    
    def pearson_correlation(self,df):
        df = df.apply(LabelEncoder().fit_transform)
        dct = {key:[value] for key, value in zip(df.corr().loc["Checkworthy",:].index, df.corr().loc["Checkworthy",:].to_list())}
        return pd.DataFrame.from_dict(dct)
    
    def chi_square(self, df):
        dct = {}
        for col in df.columns:
            data_crosstab = pd.crosstab(df['Checkworthy'], df[col])
            chi2_stat, p, dof, expected = scipy.stats.chi2_contingency(data_crosstab)
            dct[col] = [p]
        return pd.DataFrame.from_dict(dct)
    
    def correlation(self, df):
        corr_df = pd.concat([
            self.matthew_correlation(df),
            self.chi_square(df),
            self.pearson_correlation(df)
        ], ignore_index = True)
        corr_df["Measure"] = ["Matthew Correlation", "Chi Square", "Pearson Correlation"]
        cols = list(corr_df.columns)
        cols = [cols[-1]] + cols[:-1]
        return corr_df[cols]
    
    def crosstab(self):
        return pd.crosstab(self.df['Checkworthy'], 
                         self.df['Public Interest'],
                         margins = False)

    
    def save(self, binary = True):
        scale = "binary" if binary else "multiclass"
        self.correlation_df.to_csv(self.statistics_path + f'CovidCorrelation_{scale}.csv', index = False)
if __name__ == '__main__':
    cc = CovidCorrelation()
    