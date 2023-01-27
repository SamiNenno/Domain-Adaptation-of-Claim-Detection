from dataclasses import replace
import hashlib
import os
import re
from datetime import datetime
import emoji
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, \
    EarlyStoppingCallback, set_seed

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from Dataloader.TrainingSetup import DataSetup
from Utils.Evaluator import Evaluator
from codecarbon import EmissionsTracker



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def claimspotting_score(train_name, test_name, test_frame, y_true, y_probs): ## Added by Sami

    if train_name == test_name:
        test_frame = None
    evaluator = Evaluator(model = 'GermevalEnsemble',
                train = train_name,
                test = test_name,
                test_frame=None,#test_frame,
                task_type = 'checkworthy',
                y_true = y_true, 
                y_probs = y_probs,
                )
    evaluator.evaluate()

def data_dct(): ## Added by Sami
    return {
        #!'checkthat2019' : 'checkthat2019',
        #!'checkthat2021' : 'checkthat2021', 
        #!'checkthat2022' : 'checkthat2022', 
        #!'claimbuster' : 'claimbuster',
        #!'germeval' : 'germeval',
        #!'multifc' : 'multifc',
        'claimrank' : 'claimrank'
    }

def rename_for_task(df): ## Added by Sami
    df['toxic'] = df['Label']
    df['engaging'] = df['Label']
    df['fact'] = df['Label']
    df.rename(columns={'Text': 'text'}, inplace=True)
    #df.drop_duplicates(inplace=True)
    df = df.sample(frac=1, random_state=2022).reset_index(drop=True)
    df.index.rename('id', inplace=True) #? Does it need that?
    #df = df.groupby('fact').sample(n=52, random_state=2022, replace=True) #! Change back
    return df

def load_datasets(trainset:str):
    datasets = data_dct() ## Added by Sami
    datasetup = DataSetup() ## Added by Sami
    #!trainset = datasets.pop(trainset) ## Added by Sami
    df_train, df_test = datasetup.choose_one(dataset = trainset, split = True, relabel="checkworthy")
    df_train = rename_for_task(df_train)
    df_test = rename_for_task(df_test)
    df_test_dct = {trainset:{'frame':df_test}}
    for key, value in datasets.items():
        test_set = datasetup.choose_one(dataset = value, split = False, relabel="checkworthy", augmentation=False)
        test_set = rename_for_task(test_set)
        df_test_dct[key] = {'frame':test_set}
        
    # load dataset
    '''df_train = pd.read_csv('../../dataset/GermEval21_Toxic_Train.csv', index_col=0)
    df_test = pd.read_csv('../../dataset/GermEval21_Toxic_TestData.csv', index_col=0)

    # set column and index names
    df_train.rename(columns={'comment_text': 'text',
                             'Sub1_Toxic': 'toxic',
                             'Sub2_Engaging': 'engaging',
                             'Sub3_FactClaiming': 'fact'}, inplace=True)
    df_train.index.rename('id', inplace=True)

    df_test.rename(columns={'comment_text': 'text',
                            'Sub1_Toxic': 'toxic',
                            'Sub2_Engaging': 'engaging',
                            'Sub3_FactClaiming': 'fact'}, inplace=True)
    df_test.index.rename('id', inplace=True)

    # remove duplicates
    df_train.drop_duplicates(inplace=True)

    # shuffle dataset randomly
    df_train = df_train.sample(frac=1, random_state=9).reset_index(drop=True)'''

    return df_train, df_test_dct


def remove_in_word_whitespaces(comment):
    find = re.findall(r'(^| )(([a-zA-zäöüß] ){1,}[a-zA-zäöüß!?,.]([^a-zA-zäöüß]|$))', comment)
    if len(find) > 0:
        for match in find:
            found = match[0] + match[1]
            replacement = ' ' + re.sub(r' ', '', found) + ' '
            comment = comment.replace(found, replacement, 1)
    return comment


def demojize(comment):
    return emoji.demojize(comment, delimiters=(' <', '> '))


def clean_up_comments(df):
    # insert whitespaces before and after emojis so they are tokenized as separate tokens
    df['text'] = df['text'].apply(lambda t: demojize(t))
    df['text'] = df['text'].apply(lambda t: emoji.emojize(t, delimiters=('<', '>')))

    # convert terms like "a k t u e l l" to "aktuell"
    df['text'] = df['text'].apply(lambda t: remove_in_word_whitespaces(t))

    # trim mutliple whitespace characters
    df['text'] = df['text'].str.replace(r' {2,}', ' ', regex=True)

    # strip outer whitespaces
    df['text'] = df['text'].str.strip()
    return df


class GermEvalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# class MultilabelTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits
#         loss_fct = torch.nn.BCEWithLogitsLoss()
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels),
#                         labels.float().view(-1, self.model.config.num_labels))
#         return (loss, outputs) if return_outputs else loss


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    softmax = torch.nn.Softmax(dim=1)
    predictions = np.argmax(softmax(torch.tensor(logits)), axis=-1).detach().cpu().numpy()
    return {'F1': calc_f1_score_germeval(labels, predictions)}


def get_hugging_face_name(name):
    if name == 'gbert':
        return 'deepset/gbert-large'
    if name == 'gelectra':
        return 'deepset/gelectra-large'
    if name == 'gottbert':
        return 'uklfr/gottbert-base'
    if name == 'multi-roberta': ## Added by Sami
        return 'xlm-roberta-base' ## Added by Sami
    if name == 'multi-bert': ## Added by Sami
        return 'bert-base-multilingual-uncased' ## Added by Sami
    return ''


def compute_scores_for_threshold(trainer, dataset):
    s_t, s_e, s_f = np.array([]), np.array([]), np.array([])
    pred_proba = sigmoid(trainer.predict(dataset).predictions)
    for t in np.arange(0, 1.025, 0.025):
        pred = (pred_proba >= t) * 1
        s_t = np.append(s_t, calc_f1_score_germeval(dataset.labels[:, 0], pred[:, 0]))
        s_e = np.append(s_e, calc_f1_score_germeval(dataset.labels[:, 1], pred[:, 1]))
        s_f = np.append(s_f, calc_f1_score_germeval(dataset.labels[:, 2], pred[:, 2]))
    s_t = s_t.reshape((len(s_t), 1))
    s_e = s_e.reshape((len(s_e), 1))
    s_f = s_f.reshape((len(s_f), 1))
    return s_t, s_e, s_f


def calc_f1_score_germeval(ly_true, ly_pred):
    macro_f1 = 0
    if len(ly_true.shape) == 1:
        ly_true = ly_true[:, np.newaxis]
        ly_pred = ly_pred[:, np.newaxis]
    for i in range(ly_true.shape[1]):
        report = classification_report(ly_true[:, i], ly_pred[:, i], output_dict=True)
        precision_score = report['macro avg']['precision']
        recall_score = report['macro avg']['recall']
        lf1_score = 0
        if precision_score + recall_score > 0:
            lf1_score = 2 * precision_score * recall_score / (precision_score + recall_score)
        macro_f1 += lf1_score
    return macro_f1 / ly_true.shape[1]


if __name__ == '__main__':
    def fit(train_name):
        # relevant inputs
        tracker = EmissionsTracker(project_name="GermevalEnsemble", output_dir='/home/sami/Claimspotting/Metadata/Emissions/')
        tracker.start()
        model_count = 30
        model_names = ['multi-roberta', 'multi-bert']
        # model_names = ['gbert', 'gelectra', 'gottbert']
        df_train, df_test_dct = load_datasets(train_name)
        now = datetime.now()
        now = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f"Trainframe shape: {df_train.shape}\n{now}")
        df_train = clean_up_comments(df_train)
        #df_test = clean_up_comments(df_test)
        #y_test = df_test[["toxic", "engaging", "fact"]].to_numpy()

        predicted_labels = {}

        for i_label, label in enumerate(["fact"]): #["toxic", "engaging", "fact"]
            #predictions_test = []

            for i, model_name in enumerate(model_names):
                
                tokenizer = AutoTokenizer.from_pretrained(get_hugging_face_name(model_name))
                for key, value in df_test_dct.items():
                    df_test_dct[key]['predictions_test'] = []
                    df_test_dct[key]['frame'] = clean_up_comments(df_test_dct[key]['frame'])
                    y_test = df_test_dct[key]['frame'][["toxic", "engaging", "fact"]].to_numpy()
                    tokens_test = tokenizer(df_test_dct[key]['frame']['text'].tolist(), return_tensors='pt', padding='max_length',
                                        truncation=True, max_length=200)
                    df_test_dct[key]['torch_dataset'] = GermEvalDataset(tokens_test, y_test[:, i_label])
                    print(f"Frame Shape: {df_test_dct[key]['frame'].shape}")
                #tokens_test = tokenizer(df_test['text'].tolist(), return_tensors='pt', padding='max_length',
                #                        truncation=True, max_length=200)
                #dataset_test = GermEvalDataset(tokens_test, y_test[:, i_label])

                for k in range(0, model_count):
                    now = datetime.now()
                    now = now.strftime("%d/%m/%Y %H:%M:%S")
                    print(f'\n\n-------------------------------------------')
                    print(f'-------------------------------------------')
                    print(f'Train {model_name} {i+1}/{len(model_names)}')
                    print(f'Model number {k}/{model_count}')
                    print(f'- - - {now} - - -')
                    print(f'-------------------------------------------')
                    print(f'-------------------------------------------\n\n')
                    df_train_val = df_train.sample(frac=0.1, random_state=k)
                    df_train_train = df_train.drop(df_train[df_train['text'].isin(df_train_val['text'])].index)

                    tokens_train_train = tokenizer(df_train_train['text'].tolist(), return_tensors='pt',
                                                padding='max_length', truncation=True, max_length=200)
                    tokens_train_val = tokenizer(df_train_val['text'].tolist(), return_tensors='pt', padding='max_length',
                                                truncation=True, max_length=200)

                    dataset_train_train = GermEvalDataset(tokens_train_train,
                                                        df_train_train[label].to_numpy())
                    dataset_train_val = GermEvalDataset(tokens_train_val,
                                                        df_train_val[label].to_numpy())

                    hash = hashlib.sha256(pd.util.hash_pandas_object(df_train_train,
                                                                    index=True).values).hexdigest() + '_' + get_hugging_face_name(
                        model_name)[get_hugging_face_name(model_name).find('/') + 1:] + '_' + label

                    training_args = TrainingArguments(f'{model_name}_trainer',
                                                    no_cuda=False,
                                                    metric_for_best_model='F1',
                                                    load_best_model_at_end=True,
                                                    num_train_epochs=10,
                                                    eval_steps=50,
                                                    per_device_train_batch_size=24,
                                                    evaluation_strategy='steps',
                                                    seed=i * 100 + k,
                                                    learning_rate=5e-5,
                                                    warmup_ratio=0.3)

                    model = None
                    try: #
                        model = AutoModelForSequenceClassification.from_pretrained('/home/sami/Claimspotting/Models/GermevalTransformer/models/' + hash,
                                                                                local_files_only=True,
                                                                                num_labels=2)
                        trainer = Trainer(
                            model=model,
                            args=training_args,
                            train_dataset=dataset_train_train,
                            eval_dataset=dataset_train_val,
                            compute_metrics=compute_metrics,
                        )
                    except EnvironmentError:
                        set_seed(training_args.seed)
                        model = AutoModelForSequenceClassification.from_pretrained(get_hugging_face_name(model_name),
                                                                                num_labels=2)
                        trainer = Trainer(
                            model=model,
                            args=training_args,
                            train_dataset=dataset_train_train,
                            eval_dataset=dataset_train_val,
                            compute_metrics=compute_metrics,
                            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
                        )
                        trainer.train()
                        print('Model Trained!')
                        #model.save_pretrained('/home/sami/Claimspotting/Models/GermevalTransformer/models/' + hash)
                    for key, value in df_test_dct.items():
                        now = datetime.now()
                        now = now.strftime("%d/%m/%Y %H:%M:%S")
                        print(f"Compute logits for {key}\n{now}")
                        logits = trainer.predict(df_test_dct[key]['torch_dataset']).predictions
                        print(logits.shape)
                        if len(df_test_dct[key]['predictions_test']) == 0:
                            df_test_dct[key]['predictions_test'] = logits ## No Softmax
                        else:
                            df_test_dct[key]['predictions_test'] = df_test_dct[key]['predictions_test'] + logits

                        
                    #logits = trainer.predict(dataset_test).predictions
                    #softmax = torch.nn.Softmax(dim=1)
                    #pred = softmax(torch.tensor(logits)).detach().cpu().numpy()
                    #if len(predictions_test) == 0:
                    #    predictions_test = pred
                    #else:
                    #    predictions_test = predictions_test + pred
            for key, value in df_test_dct.items():
                now = datetime.now()
                now = now.strftime("%d/%m/%Y %H:%M:%S")
                print(f"Add predictions to {key}\n{now}")
                df_test_dct[key]['y_pred_proba'] = df_test_dct[key]['predictions_test'] / (model_count * len(model_names))
                df_test_dct[key]['y_pred'] = np.argmax(df_test_dct[key]['y_pred_proba'], axis=-1)
                print(f"Prediction Shape: {df_test_dct[key]['y_pred'].shape}")
                print('__________________\n')
                print(df_test_dct[key]['y_pred'])
                print('__________________\n')
            #y_pred_proba = predictions_test / (model_count * len(model_names))
            #y_pred = np.argmax(y_pred_proba, axis=-1)

            #predicted_labels[label] = y_pred

        #df_test['Sub1_Toxic'] = predicted_labels['toxic']
        #df_test['Sub2_Engaging'] = predicted_labels['engaging']
        #df_test['Sub3_FactClaiming'] = predicted_labels['fact']
        #df_test = df_test.drop(columns=['text', 'toxic', 'engaging', 'fact'])
        #df_test.index.rename('comment_id', inplace=True)
        #df_test.to_csv('results/answer.csv')
        for key, value in df_test_dct.items():
            now = datetime.now()
            now = now.strftime("%d/%m/%Y %H:%M:%S")
            print(f'Test on {key}\n{now}')
            #try:
            claimspotting_score(train_name, key, df_test_dct[key]['frame'], df_test_dct[key]['frame'][["fact"]].to_numpy().reshape(-1), df_test_dct[key]['y_pred_proba'])
            now = datetime.now()
            now = now.strftime("%d/%m/%Y %H:%M:%S")
            print(f'Model succesfully tested on {key}!\n{now}')
            #except Exception as e:
            #    now = datetime.now()
            #    now = now.strftime("%d/%m/%Y %H:%M:%S")
            #    print(f'Model not succesfully tested on {key}!\n{now}')
            #    print(f'Error: {e}')
        tracker.stop() 
        ''' with open('results/scores.txt', 'w') as f:
            f.write(f'F1 score for class toxic: {calc_f1_score_germeval(y_test[:, 0], predicted_labels["toxic"])}\n')
            f.write(f'F1 score for class engaging: {calc_f1_score_germeval(y_test[:, 1], predicted_labels["engaging"])}\n')
            f.write(f'F1 score for class fact-claiming: {calc_f1_score_germeval(y_test[:, 2], predicted_labels["fact"])}\n')
            y_pred_all = np.c_[predicted_labels["toxic"], predicted_labels["engaging"], predicted_labels["fact"]]
            f.write(f'macro F1 score: {calc_f1_score_germeval(y_test, y_pred_all)}\n') '''
    for idx, train_name in enumerate(['checkthat2019', 'checkthat2021', 'checkthat2022']): #claimbuster, 'germeval', 'claimrank',
        print(f'\n\nTrain on {train_name}\n\n')
        fit(train_name)
        
