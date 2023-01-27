import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
import random
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
import json
import emoji
from datetime import datetime
import torch
import gc
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import evaluate
from scipy.special import softmax
from codecarbon import EmissionsTracker
from Utils.Evaluator import Evaluator
from Dataloader.TrainingSetup import DataSetup
np.random.seed(2022)
random.seed(2022)



class TransformerTester(DataSetup):
    def __init__(self, stats) -> None:
        super().__init__()
        self.stats = stats
        self.config = AutoConfig.from_pretrained(self.stats["model"], num_labels=self.stats["labels"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.stats["model"])
        self.log_path = self.model_path + 'Transformer/'
        
        self.random_seed = 2022
        with open(self.metadata_path + 'TransformerTestProtocoll.json', 'r') as f: 
            self.protocoll = json.load(f)

    def ValSplit(self, df, frac:float = 0.9):
        df_big = df.groupby("label").sample(frac=frac, random_state = self.random_seed)
        df_small = df.drop(df_big.index)
        return df_big.reset_index(drop=True), df_small.reset_index(drop=True) 
    
    def replace_emoji(self, df):
        df['text'] = df['text'].apply(lambda sentence: emoji.demojize(sentence))
        return df

    def get_train_data(self):
        train_frame, test_frame = self.choose_one(dataset = self.stats['dataset'], split = True, relabel = self.stats['task'], augmentation = True)
        train_frame = train_frame.rename(columns={"Text": "text", "Label": "label"})
        test_frame = test_frame.rename(columns={"Text": "text", "Label": "label"})
        #if self.stats['dataset'] in ['checkthat2022', 'germeval']:
        #    train_frame, test_frame = self.replace_emoji(train_frame), self.replace_emoji(test_frame)
        train_frame, val_frame = self.ValSplit(train_frame)
        self.trainset = self.turn_to_dataset(train_frame)
        self.valset = self.turn_to_dataset(val_frame)
        self.testset = self.turn_to_dataset(test_frame)
        self.test_frame = test_frame

    def turn_to_dataset(self, frame):
        def encode(examples):
            outputs = self.tokenizer(examples['text'], truncation=True)
            return outputs
        return Dataset.from_pandas(frame).map(encode, batched=False) 
    
    def get_model(self):
        return AutoModelForSequenceClassification.from_pretrained(self.stats["model"], config = self.config)

    def build_trainer(self):
        self.get_train_data()
        
        def build_compute_metrics_fn(stats:dict):
            m = 'accuracy' if stats['eval_metric'] == 'eval_acc' else 'f1'
            def compute_metrics(eval_pred):
                metric = evaluate.load(m)
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                if m == 'accuracy':
                    return metric.compute(predictions=predictions, references=labels)
                else:
                    return metric.compute(predictions=predictions, references=labels, average='macro')
            return compute_metrics

        training_args = TrainingArguments(
                output_dir=self.log_path,
                learning_rate=self.stats['learning_rate'],
                do_train=True,
                do_eval=True,
                no_cuda=False,
                evaluation_strategy="epoch",
                save_strategy="no",
                load_best_model_at_end=False,
                num_train_epochs=self.stats['num_train_epochs'],
                max_steps=-1,
                per_device_train_batch_size=self.stats['per_device_train_batch_size'],
                warmup_steps=0,
                weight_decay=self.stats['weight_decay'],
                logging_dir=self.log_path + "logs",
                skip_memory_metrics=True,
                report_to="none",
                )
        trainer = Trainer(
                model_init=self.get_model,
                tokenizer=self.tokenizer,
                args=training_args,
                train_dataset=self.trainset,
                eval_dataset=self.valset ,
                compute_metrics=build_compute_metrics_fn(self.stats),
                )
        return trainer

    def fit(self):
        self.trainer = self.build_trainer()
        self.trainer.train()
        self.save_model()
        y_probs = softmax(self.trainer.predict(self.testset).predictions, axis=1)
        y_true = np.array(self.test_frame['label'].to_list())
        evaluator = Evaluator(model = self.stats["model"],
                test_frame = None,
                train = self.stats["dataset"],
                test = self.stats["dataset"],
                task_type = self.stats["task"],
                y_true = y_true, 
                y_probs = y_probs,
                )
        evaluator.evaluate()

    def cross_test(self):
        for idx, X in enumerate(self.protocoll[self.stats["dataset"]][self.stats["task"]]):
            now = datetime.now()
            now = now.strftime("%d/%m/%Y %H:%M:%S")
            print("\n\n---------------------------")
            print(f"Testrun: {idx+1}/{len(self.protocoll[self.stats['dataset']][self.stats['task']])}")
            print(f"Time: {now}")
            print(f"Trainset: {self.stats['dataset']}")
            print(f"Testset: {X}")
            print(f"Task Type: {self.stats['task']}")
            print("---------------------------\n\n")
            self.test(dataset = X, task = self.stats["task"])

    def test(self, dataset, task):
        df = self.choose_one(dataset=dataset, split=False, relabel=task, augmentation=False)
        df = df.rename(columns={"Text": "text", "Label": "label"})
        test_data = self.turn_to_dataset(df)
        y_probs = softmax(self.trainer.predict(test_data).predictions, axis=1)
        y_true = np.array(df['label'].to_list())
        evaluator = Evaluator(model = self.stats["model"],
                test_frame=df,
                train = self.stats["dataset"],
                test = dataset,
                task_type = task,
                y_true = y_true, 
                y_probs = y_probs,
                )
        evaluator.evaluate()
        try:
            print(json.dumps(evaluator.metric_dct, sort_keys=False, indent=6))
        except Exception:
            pass

    def save_model(self):
        path = self.log_path
        if not os.path.exists(path):
            os.makedirs(path) 
        path += 'FinalModels' + '/'
        if not os.path.exists(path):
            os.makedirs(path) 
        path += self.stats['dataset'] + '/'
        if not os.path.exists(path):
            os.makedirs(path) 
        path += self.stats['task'] + '/'
        if not os.path.exists(path):
            os.makedirs(path) 
        self.trainer.save_model(path + self.stats['model'])
        self.tokenizer.save_pretrained(path + 'tokenizer_' + self.stats['model'])
    
    def execute(self):
        tracker = EmissionsTracker(project_name="Transformer_Testing", output_dir=self.emission_path)
        tracker.start()
        gc.collect()
        torch.cuda.empty_cache()
        self.fit()
        gc.collect()
        torch.cuda.empty_cache()
        self.cross_test()
        tracker.stop()  


        
if __name__ == '__main__':
    '''print('Start Testing!')
    dir = '/home/sami/Claimspotting/Metadata/Transformer_Hypparam/multilingual/checkworthy/'
    for dataset in os.listdir(dir):
        now = datetime.now()
        now = now.strftime("%d/%m/%Y %H:%M:%S")
        print('\n\n\n-----------------------------------')
        print('-----------------------------------')
        print('-----------------------------------')
        print(f'Train on {dataset}')
        print(f"Time: {now}")
        print('-----------------------------------')
        print('-----------------------------------')
        print('-----------------------------------\n\n\n')
        with open(dir + dataset + '/HyperParam.json', 'r') as f: 
            hypparamdict = json.load(f)
        print(json.dumps(hypparamdict, sort_keys=False, indent=6))
        try:
            tt = TransformerTester(hypparamdict)
            tt.execute()
        except Exception as e:
            print(e)'''
            
    ger_path = '/home/sami/Claimspotting/Metadata/Transformer_Hypparam/multilingual/checkworthy/germeval/HyperParam.json'
    with open(ger_path, 'r') as f: 
            hypparamdict = json.load(f)
    print(json.dumps(hypparamdict, sort_keys=False, indent=6))
    try:
        tt = TransformerTester(hypparamdict)
        tt.execute()
    except Exception as e:
        print(e)
    
    
    
        
    
