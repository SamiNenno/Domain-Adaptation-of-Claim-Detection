from TransformerUtils import get_data_and_stats, get_path
import os
import json
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.stopper import TrialPlateauStopper

from transformers import AutoModelForSequenceClassification, EvalPrediction, TrainingArguments, Trainer, AutoTokenizer
from typing import Callable, Dict
from sklearn.metrics import f1_score
import numpy as np
from codecarbon import EmissionsTracker
from datetime import datetime


class CustomTrainer():
        def __init__(self,
                model_language:str,
                model_size:str,
                datasetname:str,
                relabel:str,
                num_samples:int = 5,
                max_iter:int = 4, 
                ):
            self.data, self.stats = get_data_and_stats(model_language, model_size, datasetname, relabel)
            self.eval_metric = 'eval_acc' if not 'checkthat' in datasetname else 'eval_f1'
            self.stats['eval_metric'] = self.eval_metric
            self.num_samples = num_samples
            self.tokenizer = AutoTokenizer.from_pretrained(self.stats['model'], use_fast=True)
            self.metadata_path, self.model_path, self.emission_path = get_path()
            self.update_path(datasetname, relabel, model_language)

        def update_path(self, datasetname, relabel, model_language):
            self.hypparam_path = self.metadata_path + "Transformer_Hypparam/"
            self.model_path += 'Transformer/'
            self.model_path = self.create_directories(self.model_path, datasetname, relabel, model_language)
            self.hypparam_path = self.create_directories(self.hypparam_path, datasetname, relabel, model_language)
            self.hypparam_path += 'HyperParam.json'

        def create_directories(self, path, datasetname, relabel, model_language):
            if not os.path.exists(path):
                os.makedirs(path) 
            path += model_language + '/'
            if not os.path.exists(path):
                os.makedirs(path) 
            path += relabel + '/'
            if not os.path.exists(path):
                os.makedirs(path) 
            path += datasetname + '/'
            if not os.path.exists(path):
                os.makedirs(path) 
            return path

        def get_model(self):
                return AutoModelForSequenceClassification.from_pretrained(self.stats["model"], num_labels=self.stats["labels"])

        def set_up_trainer(self):
                def build_compute_metrics_fn(eval_acc:str) -> Callable[[EvalPrediction], Dict]:
                    def simple_accuracy(y_true, y_pred):
                                return (y_pred == y_true).mean()
                    def macro_f1(y_true, y_pred):
                        return f1_score(y_true, y_pred, average='macro')
                    metric = simple_accuracy if eval_acc == 'eval_acc' else macro_f1
                    name = 'acc' if eval_acc == 'eval_acc' else 'f1'
                    def compute_metrics_fn(p: EvalPrediction):
                            preds = np.argmax(p.predictions, axis=1)
                            metrics = {name: metric(y_true = p.label_ids, y_pred = preds)}
                            return metrics
                    return compute_metrics_fn

                training_args = TrainingArguments(
                output_dir=self.model_path,
                learning_rate=1e-5,  # config
                do_train=True,
                do_eval=True,
                no_cuda=False,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                num_train_epochs=2,  # config
                max_steps=-1,
                per_device_train_batch_size=16,  # config
                per_device_eval_batch_size=16,  # config
                warmup_steps=0,
                weight_decay=0.1,  # config
                logging_dir=self.model_path,#"./logs",
                skip_memory_metrics=True,
                report_to="none",
                )
                
                trainer = Trainer( 
                model_init=self.get_model,
                tokenizer=self.tokenizer,
                args=training_args,
                train_dataset=self.data['train'],
                eval_dataset=self.data['val'],
                compute_metrics=build_compute_metrics_fn(self.eval_metric),
                )
                return trainer

        def hyperparamter_tuning(self):
                ray.init(ignore_reinit_error=True)
                trainer = self.set_up_trainer()
                tune_config = {
                "per_device_train_batch_size": tune.choice([16, 32, 64]),
                "per_device_eval_batch_size": 32,
                "num_train_epochs": tune.choice([3, 4, 5, 6, 7]),
                "max_steps": -1, }

                scheduler = PopulationBasedTraining(
                time_attr="training_iteration",
                metric=self.eval_metric,
                mode="max",
                perturbation_interval=1,#2?
                hyperparam_mutations={
                        "weight_decay": tune.uniform(0.0, 0.3),
                        "learning_rate": tune.uniform(1e-5, 5e-5),
                        "per_device_train_batch_size": [16, 32, 64],},)

                reporter = CLIReporter(
                parameter_columns={
                        "weight_decay": "w_decay",
                        "learning_rate": "lr",
                        "per_device_train_batch_size": "train_bs/gpu",
                        "num_train_epochs": "num_epochs",
                },
                metric_columns=[self.eval_metric, "eval_loss", "epoch", "training_iteration"],
                max_report_frequency = 70
                )
                best_trial = trainer.hyperparameter_search(
                hp_space=lambda _: tune_config,
                backend="ray",
                n_trials=self.num_samples,
                resources_per_trial={"cpu": 10, "gpu": 1},
                scheduler=scheduler,
                keep_checkpoints_num=1,
                checkpoint_score_attr="training_iteration",
                stop=TrialPlateauStopper(metric=self.eval_metric),
                progress_reporter=reporter,
                local_dir= self.model_path + "/ray_results/", 
                name="tune_transformer_pbt",
                log_to_file=True,
                )
                return best_trial

        def train(self):
                tracker = EmissionsTracker(project_name="Transformer_Training", output_dir=self.metadata_path)
                tracker.start()
                best_trial = self.hyperparamter_tuning()
                self.stats = self.stats | best_trial.hyperparameters # Works only in Python 3.9+
                with open(self.hypparam_path, 'w') as json_file:
                    json.dump(self.stats, json_file, indent=6)
                tracker.stop() 
if __name__ == '__main__':
    datasets = ['germeval','checkthat2019', 'checkthat2021', 'checkthat2022', 'claimbuster', 'claimrank']
    for idx, dataset in enumerate(datasets): 
        now = datetime.now()
        now = now.strftime("%d/%m/%Y %H:%M:%S")
        print('\n\n\n-----------------------------------')
        print('-----------------------------------')
        print(f"Model: Transformer") 
        print(f"Time: {now}")
        print(f"Fit model {idx+1}/{len(datasets)}:")
        print(f"Training Data: {dataset}")     
        print('-----------------------------------')
        print('-----------------------------------\n\n\n')
        trainer = CustomTrainer("multilingual", "base", dataset, "checkworthy")
        trainer.train()
