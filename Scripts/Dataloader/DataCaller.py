from pydoc import describe
from tqdm import tqdm
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from Utils.Ressources import Ressources
from Dataloader.Claimbuster_Loader import ClaimbusterLoader
from Dataloader.Checkthat2019_Loader import Checkthat2019Loader
from Dataloader.Checkthat2021_Loader import Checkthat2021Loader
from Dataloader.Checkthat2022_Loader import Checkthat2022Loader
from Dataloader.Claimrank_Loader import ClaimrankLoader
from Dataloader.GermEval_Loader import GermEvalLoader
from Dataloader.MultiFC_Loader import MultiFCLoader

class DataCaller(Ressources):
    def __init__(self):
        self.load_paths()
        self.load_json()
        self.random_seed = 2022
        self.loader = {
            "claimbuster" : ClaimbusterLoader,
            "checkthat2019" : Checkthat2019Loader,
            "checkthat2021" : Checkthat2021Loader,
            "checkthat2022" : Checkthat2022Loader,
            "claimrank" : ClaimrankLoader,
            "germeval" : GermEvalLoader,
            "multifc" : MultiFCLoader
        }
        self.all = self.loader.keys() 
        self.factchecking = ["claimbuster", "checkthat2019", "checkthat2021", "checkthat2022", "claimrank","germeval", "multifc"]
        self.argument = []
    
    def list_all_loaders(self):
        for l in self.loader.keys():
            print(l)
            
    def select(self, loaders:list = ["all"]):
        temp = {}
        if len(loaders) == 1:
            if loaders[0].lower() == "all":
                selection = self.all
            elif loaders[0].lower() == "factchecking":
                selection = self.factchecking
            elif loaders[0].lower() == "argument":
                selection = self.argument
            elif loaders[0] in self.loader.keys():
                selection = loaders
            else:
                raise ValueError(f"{loaders[0]} is not an option. Chose 'all', 'factchecking', 'argument', or name specific loaders!")
        else:
            selection = []
            for loader in loaders:
                if loader in self.loader.keys():
                    selection.append(loader)
                else:
                    raise ValueError(f"{loader} is not among the available loaders. Try one or more of the following:\n{list(self.all)}!")
              
        for selected in tqdm(selection, desc = "Load Datasets"):
            if selected in self.all:
                temp[selected] = self.loader[selected]()
        self.loader = temp
    
    def get_single_loader(self, loader:str):
        return self.loader[loader]
        
    def get_all_data(self):
        return self.loader.values()
        
    def get_multi_label_data(self):
        return [self.loader["claimbuster"],
                self.loader["checkthat2019"],
                self.loader["checkthat2021"],
                self.loader["checkthat2022"],
                self.loader["claimrank"],
                self.loader["germeval"]]

if __name__ == '__main__':
    dc = DataCaller()
    dc.select(["all"])
    print(dc.get_single_loader("claimbuster"))
    
        
    