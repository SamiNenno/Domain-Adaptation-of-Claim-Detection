import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import xgboost as xgb
from ray import tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import CLIReporter
from ray.tune.stopper import TrialPlateauStopper
from codecarbon import EmissionsTracker
import torch
import gc
np.random.seed(2022)
random.seed(2022)



def train_test_split(root:str, dataset:str, augmentation:bool):
    train_path = f"{root}Datasets/SentenceEmbeddings/{'augmented' if augmentation else 'not_augmented'}/{dataset}_train.csv"
    test_path = f"{root}Datasets/SentenceEmbeddings/{'augmented' if augmentation else 'not_augmented'}/{dataset}_test.csv"
    train_frame, test_frame = pd.read_csv(train_path), pd.read_csv(test_path)
    train_frame, val_frame = ValSplit(train_frame)
    train_x, val_x, test_x = train_frame.iloc[:,2:].to_numpy(), val_frame.iloc[:,2:].to_numpy(), test_frame.iloc[:,2:].to_numpy()
    train_y, val_y, test_y = train_frame['Label'].to_numpy(), val_frame['Label'].to_numpy(), test_frame['Label'].to_numpy()
    train_set = xgb.DMatrix(train_x, label=train_y)
    val_set = xgb.DMatrix(val_x, label=val_y)
    test_set = xgb.DMatrix(test_x, label=test_y)
    ## https://xgboost.readthedocs.io/en/stable/python/examples/feature_weights.html
    return train_set, val_set, test_set

def ValSplit(df, frac:float = 0.9):
        df_big = df.groupby("Label").sample(frac=frac, random_state = 2022)
        df_small = df.drop(df_big.index)
        return df_big.reset_index(drop=True), df_small.reset_index(drop=True) 

def create_dir(root:str, dataset:str):
    result_path = root + 'Models/XGBoost/'
    if not os.path.exists(result_path):
        os.makedirs(result_path) 
    result_path += f"{dataset}/"
    if not os.path.exists(result_path):
        os.makedirs(result_path) 
    checkpoint_path = result_path + 'checkpoints/'
    return result_path, checkpoint_path
    

def tune_xgboost(result_path:str, dataset:str, augmentation:bool, log_dir:str, trial:int=50):
    def train_claimspotting(config):
        np.random.seed(2022)
        random.seed(2022)
        train_set, val_set, test_set = train_test_split(root=root, dataset=dataset, augmentation=augmentation)
        xgb.train(
            config,
            train_set,
            evals=[(val_set, "eval")], 
            verbose_eval=False,
            callbacks=[TuneReportCheckpointCallback(filename=f"{result_path}model.xgb")],
            num_boost_round = 5000)

    search_space = {
        # You can mix constants with search space objects.
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "tree_method": "gpu_hist",
        "max_depth": tune.randint(2, 9),
        "min_child_weight": tune.choice([1, 2, 3]),
        "subsample": tune.uniform(0.5, 0.8),
        "eta": tune.loguniform(1e-5, 0.9),
        "gamma" : tune.uniform(1, 9), ##! added
    }    
    pbt_scheduler =  PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=2,
            hyperparam_mutations={
        "max_depth": tune.randint(2, 9),
        "min_child_weight": tune.choice([1, 2, 3]),
        "subsample": tune.uniform(0.5, 0.8),
        "eta": tune.loguniform(1e-4, 0.9),
        "gamma" : tune.uniform(1, 9), ##! added
        })
    
    reporter = CLIReporter(
                parameter_columns={
                        "max_depth": "max_depth",
                        "min_child_weight": "min_child_weight",
                        "subsample": "subsample/gpu",
                        "eta": "eta",
                        "gamma" : "gamma",##! added
                },
                metric_columns=["eval-logloss", "training_iteration"],
                max_report_frequency = 10
                )
    analysis = tune.run(
        train_claimspotting,
        metric="eval-logloss",
        mode="min",
        # You can add "gpu": 0.1 to allocate GPUs
        resources_per_trial={"cpu": 25, "gpu":1/trial},
        config=search_space,
        num_samples=trial,
        scheduler=pbt_scheduler,
        progress_reporter=reporter,
        local_dir=log_dir,
        stop=TrialPlateauStopper(metric='eval-logloss'),)
    return analysis

def get_best_model_checkpoint(analysis, result_path):
    best_bst = xgb.Booster()
    best_bst.load_model(f"{result_path}model.xgb")
    accuracy = 1. - analysis.best_result["eval-error"]
    print(f"Best model parameters: {analysis.best_config}")
    print(f"Best model total accuracy: {accuracy:.4f}")
    with open(f"{result_path}Hypparam.json", 'w') as json_file:
        json.dump(analysis.best_config, json_file, indent=6)
    return best_bst

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    root = '/home/sami/Claimspotting/' ##! Set this to Claimspotting folder path
    datasets = ["claimbuster", "checkthat2019", "checkthat2021", "checkthat2022", "claimrank", "germeval"]
    try:
        dataset = datasets[1] # 3 war schon
        augmentation = False
        trials = 8
        result_path, checkpoint_path = create_dir(root=root, dataset=dataset)
        tracker = EmissionsTracker(project_name="XGBoost_Training", output_dir=f"{root}Metadata/Emissions/Baseline/")
        tracker.start()
        analysis = tune_xgboost(result_path = result_path, dataset = dataset, augmentation = augmentation, log_dir = checkpoint_path, trial=trials)
        best_bst = get_best_model_checkpoint(analysis, result_path=result_path)
        tracker.stop()
        del analysis
        del best_bst
        gc.collect()
        torch.cuda.empty_cache()
        #Best model total accuracy: 0.8076 for 5000 trees
        #Best model total accuracy: 0.8246 for 5000 trees
    except Exception:
        del analysis
        del best_bst
        gc.collect()
        torch.cuda.empty_cache()
    