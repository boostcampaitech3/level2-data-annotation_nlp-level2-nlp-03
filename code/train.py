import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
from custom.trainer import customTrainer3
from custom.callback import customWandbCallback
import wandb
from sklearn.metrics import confusion_matrix
from utils import plot_cm_by_num_samples, plot_cm_by_ratio
import argparse

label_list = [
    'no_relation',
    'dat:alter_name',
    'dat:feature',
    'dat:influence',

    'idv:alter_name',
    'idv:feature',
    'idv:location',
    'idv:parent_con',
    'idv:influence',

    'phe:alter_name',
    'phe:feature',
    'phe:location',
    'phe:parent_con',
    'phe:influence',

    'res:feature',
    'res:location',
    'res:parent_con',
    'res:influence',
    'res:outbreak_date',
    'res:alter_name'
]

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_lists = [
    'no_relation',
    'dat:alter_name',
    'dat:feature',
    'dat:influence',

    'idv:alter_name',
    'idv:feature',
    'idv:location',
    'idv:parent_con',
    'idv:influence',

    'phe:alter_name',
    'phe:feature',
    'phe:location',
    'phe:parent_con',
    'phe:influence',

    'res:feature',
    'res:location',
    'res:parent_con',
    'res:influence',
    'res:outbreak_date',
    'res:alter_name'
    ]
    no_relation_label_idx = label_lists.index("no_relation")
    label_indices = list(range(len(label_lists)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels,num_classes = 20):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(num_classes)[labels]

    score = np.zeros((num_classes,))
    for c in range(num_classes):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.
  conf = confusion_matrix(labels, preds)
  fig1 = plot_cm_by_num_samples(conf, label_list)
  fig2 = plot_cm_by_ratio(conf, label_list)
  return {
      'micro f1 score': f1,
      'auprc': auprc,
      'accuracy': acc,
      'cm_samples': fig1,
      'cm_ratio': fig2
  }

def label_to_num(label):
  num_label = []
  with open('nlp3_dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def train(model_name,args):
  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"
  MODEL_NAME = model_name #'klue/roberata-small' #"klue/bert-base"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  # train_dataset = load_data("/opt/ml/test_data/nlp3_all_data.csv")
  # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

  # train_label = label_to_num(train_dataset['label'].values)
  # dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  train_data_dir = "./data/nlp3_all_data_new.csv"
  train_dataset, dev_dataset = load_split_data(train_data_dir,
                                                42,
                                                eval_ratio=0.2)

  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)

  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 20

  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  print(model.config)
  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=15,              # total number of training epochs
    learning_rate=args.lr,               # learning_rate
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=500,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 100,            # evaluation step.
    load_best_model_at_end = True,
      report_to='wandb',
      run_name=f'nature_{model_name}'
  )
  trainer = customTrainer3(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,         # define metrics function
    callbacks = [customWandbCallback()]
  )

  # train model
  trainer.train()
  model.save_pretrained('./best_model')
def main(model, args):
    wandb.login()
    wandb.init(project='nature', name=f'nature+{model}+{args.lr}', entity='kimcando')
    train(model,args)

    # models = ['klue/bert-base','bert-base-multilingual-uncased', 'klue/roberta-small', 'klue/roberta-base', 'klue/roberta-large']
    # for model in models:
        # wandb.login()
        # wandb.init(project='nature', name=f'nature+{model}', entity='kimcando')
        # train(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument('--model_name', type=str, default="./best_model")
    parser.add_argument('--lr', type=float, default=5e-5)
    args = parser.parse_args()

    main(args.model_name, args)
