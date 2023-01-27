import os
import sys
from datetime import datetime
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from BaselineScripts.Baseline import Baseline
from Utils.Ressources import Ressources


class BaselineProtocoll(Ressources):
    def __init__(self) -> None:
        self.load_paths()
        self.load_json()
        self.result_path = self.result_path + "Baseline/"
        self.data = ["claimbuster", "checkthat2019", "checkthat2021", "checkthat2022", "claimrank", "germeval", "multifc"]
        self.protocol = [
            
            ['claimrank', 'claimrank', 'checkworthy', 3] ,
            ['claimrank', 'checkthat2019', 'checkworthy', 3] ,
            ['claimrank', 'checkthat2021', 'checkworthy', 3] ,
            ['claimrank', 'checkthat2022', 'checkworthy', 3] ,
            ['claimrank', 'multifc', 'checkworthy', 3] ,
            ['claimrank', 'claimbuster', 'checkworthy', 3] ,
            
            ['claimbuster', 'claimrank', 'checkworthy', 3] ,
            ['claimbuster', 'checkthat2019', 'checkworthy', 3] ,
            ['claimbuster', 'checkthat2021', 'checkworthy', 3] ,
            ['claimbuster', 'checkthat2022', 'checkworthy', 3] ,
            ['claimbuster', 'multifc', 'checkworthy', 3] ,
            ['claimbuster', 'claimbuster', 'checkworthy', 3] ,
            
            ['checkthat2019', 'claimrank', 'checkworthy', 3] ,
            ['checkthat2019', 'checkthat2019', 'checkworthy', 3] ,
            ['checkthat2019', 'claimbuster', 'checkworthy', 3] ,
            ['checkthat2019', 'checkthat2021', 'checkworthy', 3] ,
            ['checkthat2019', 'checkthat2022', 'checkworthy', 3] ,
            ['checkthat2019', 'multifc', 'checkworthy', 3] ,

            ['checkthat2021', 'claimrank', 'checkworthy', 3] ,
            ['checkthat2021', 'claimbuster', 'checkworthy', 3] ,
            ['checkthat2021', 'checkthat2019', 'checkworthy', 3] ,
            ['checkthat2021', 'checkthat2022', 'checkworthy', 3] ,
            ['checkthat2021', 'multifc', 'checkworthy', 3] ,
            ['checkthat2021', 'checkthat2021', 'checkworthy', 3] ,

            ['checkthat2022', 'claimrank', 'checkworthy', 3] ,
            ['checkthat2022', 'checkthat2022', 'checkworthy', 3] ,
            ['checkthat2022', 'claimbuster', 'checkworthy', 3] ,
            ['checkthat2022', 'checkthat2019', 'checkworthy', 3] ,
            ['checkthat2022', 'checkthat2021', 'checkworthy', 3] ,
            ['checkthat2022', 'multifc', 'checkworthy', 3] ,

            ]
    
    def execute_protocoll(self):
        for run in range(2): 
            if run == 1: ##! REMOVE
                continue ##! REMOVE
            modelname = 'LogReg' if run == 0 else 'SVM'
            name_to_print = "Logistic Regression" if run == 0 else 'Support Vector Machine'

            for idx, p in enumerate(self.protocol):
                now = datetime.now()
                now = now.strftime("%d/%m/%Y %H:%M:%S")
                print('\n\n\n-----------------------------------')
                print('-----------------------------------')
                print(f"Model: {name_to_print}") 
                print(f"Time: {now}")
                print(f"Fit model {idx+1}/{len(self.protocol)}:")
                print(f"Training Data: {p[0]}")     
                print(f"Test Data: {p[1]}")
                print(f"Task Type: {p[2]}")
                print(f"{p[3]}-fold Cross-Validation")
                print('-----------------------------------')
                print('-----------------------------------\n\n\n')
                try:
                    baseline_model = Baseline(
                        modelname = modelname, 
                        train_name = p[0], 
                        test_name = p[1], 
                        relabel = p[2],
                        n_splits = p[3]
                    )
                    baseline_model.fit()
                    now = datetime.now()
                    now = now.strftime("%d/%m/%Y %H:%M:%S")
                    print('\n\n\n-----------------------------------')
                    print('-----------------------------------')
                    print("Training and Testing Succesful!")
                    print(f"Time: {now}")
                    print('-----------------------------------')
                    print('-----------------------------------\n\n\n')
                except Exception as e:
                    self.document_failure({
                        "Modelname": modelname, 
                        "Trainset" : p[1],
                        "Testset": p[2],
                        "Relabel":p[3],
                        "n_splits":p[4],
                        "Error":str(e)
                    })
                    print('\n\n\n-----------------------------------')
                    print('-----------------------------------')
                    print("Training and Testing NOT Succesful!")
                    print(f"Error: {e}")
                    print('-----------------------------------')
                    print('-----------------------------------\n\n\n')

if __name__ == '__main__':
    p = BaselineProtocoll()
    p.execute_protocoll()

    
