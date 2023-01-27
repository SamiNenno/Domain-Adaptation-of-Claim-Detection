import pandas as pd
from matplotlib import pyplot 
from matplotlib_venn import venn2, venn3
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from Dataloader.DataCaller import DataCaller

class VennDiagramm(DataCaller):
    def __init__(self):
        super().__init__()
        self.select(["claimbuster", "checkthat2019", "checkthat2021", "claimrank"])
    
    def create_venn(self):
        plt = venn3([
            set(self.get_single_loader('claimbuster').get_data()["Text"].to_list()), 
            set(self.get_single_loader('checkthat2019').get_data()["Text"].to_list()), 
            set(self.get_single_loader('checkthat2021').get_data()["Text"].to_list()), 
            ],
       set_labels=("Claimbuster", "Checkthat2019", "Checkthat2021")
        )
        pyplot.savefig(self.visuals_path + 'PNG/VennDiagrams/CBC19C21.png')
        pyplot.clf()
        plt = venn2([
            set(self.get_single_loader('claimbuster').get_data()["Text"].to_list()), 
            set(self.get_single_loader('checkthat2019').get_data()["Text"].to_list()), 
            ],
       set_labels=("Claimbuster", "Checkthat2019")
        )
        pyplot.savefig(self.visuals_path + 'PNG/VennDiagrams/CBC19.png')
        pyplot.clf()
        plt = venn2([
            set(self.get_single_loader('claimbuster').get_data()["Text"].to_list()), 
            set(self.get_single_loader('checkthat2021').get_data()["Text"].to_list()), 
            ],
       set_labels=("Claimbuster", "Checkthat2021")
        )
        pyplot.savefig(self.visuals_path + 'PNG/VennDiagrams/CBC21.png')
        pyplot.clf()
        plt = venn2([
            set(self.get_single_loader('checkthat2019').get_data()["Text"].to_list()), 
            set(self.get_single_loader('checkthat2021').get_data()["Text"].to_list()), 
            ],
       set_labels=("Checkthat2019", "Checkthat2021")
        )
        pyplot.savefig(self.visuals_path + 'PNG/VennDiagrams/C19C21.png')
        # -----------
        pyplot.clf()
        plt = venn2([
            set(self.get_single_loader('claimbuster').get_data()["Text"].to_list()), 
            set(self.get_single_loader('claimrank').get_data()["Text"].to_list()), 
            ],
       set_labels=("Claimbuster", "claimrank")
        )
        pyplot.savefig(self.visuals_path + 'PNG/VennDiagrams/CBCR.png')
        pyplot.clf()
        plt = venn2([
            set(self.get_single_loader('claimrank').get_data()["Text"].to_list()), 
            set(self.get_single_loader('checkthat2021').get_data()["Text"].to_list()), 
            ],
       set_labels=("claimrank", "Checkthat2021")
        )
        pyplot.savefig(self.visuals_path + 'PNG/VennDiagrams/CRC21.png')
        pyplot.clf()
        plt = venn2([
            set(self.get_single_loader('checkthat2019').get_data()["Text"].to_list()), 
            set(self.get_single_loader('claimrank').get_data()["Text"].to_list()), 
            ],
       set_labels=("Checkthat2019", "claimrank")
        )
        pyplot.savefig(self.visuals_path + 'PNG/VennDiagrams/C19CR.png')
       

   
        
if __name__ == '__main__':
    vd = VennDiagramm()
    vd.create_venn()