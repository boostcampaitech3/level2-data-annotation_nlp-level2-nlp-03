import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def load_split_data(dataset_dir, seed=42, eval_ratio=0.2):
    """ csv 파일을 경로에 맡게 불러오고,
    duplicated cls를 고려하지 않고  train, eval에 맞게 분리해줍니다
    """
    print('### Split basic')
    pd_dataset = pd.read_csv(dataset_dir)
    label = pd_dataset['label']
    pd_train, pd_eval = train_test_split(pd_dataset, test_size=eval_ratio, shuffle=True,
                                         stratify=label,
                                         random_state=seed)

    train_dataset = preprocessing_dataset(pd_train)
    eval_dataset = preprocessing_dataset(pd_eval)

    return train_dataset, eval_dataset


def label_to_num(label):
    num_label = []
    with open('nlp3_dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label



def load_split_eunki_data(dataset_dir, sedd=42, eval_ratio=0.2):
    """
    은기님 구현 로직대로 우선 StratifiedKFold 중 1개만 선택
    validation 에 있는 문장이 train에 나온다면 맞바꿔서 개별 중복 문장은 training 또는 validation에만 포함되게 함
    """
    from sklearn.model_selection import StratifiedKFold

    pd_dataset = pd.read_csv(dataset_dir)
    total_train_dataset = preprocessing_dataset(pd_dataset)

    total_train_dataset['is_duplicated'] = total_train_dataset['sentence'].duplicated(keep=False)
    result = label_to_num(total_train_dataset['label'].values)

    total_train_label = pd.DataFrame(data=result, columns=['label'])
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = list(kfold.split(total_train_dataset, total_train_label))[0]

    train_dataset = total_train_dataset.loc[train_idx]
    val_dataset = total_train_dataset.loc[val_idx]
    train_label = total_train_label.loc[train_idx]
    val_label = total_train_label.loc[val_idx]

    train_dataset.reset_index(drop=True, inplace=True)
    val_dataset.reset_index(drop=True, inplace=True)
    train_label.reset_index(drop=True, inplace=True)
    val_label.reset_index(drop=True, inplace=True)

    temp = []

    for val_idx in val_dataset.index:
        if val_dataset['is_duplicated'].iloc[val_idx] == True:
            if val_dataset['sentence'].iloc[val_idx] in train_dataset['sentence'].values:
                train_dataset.append(val_dataset.iloc[val_idx])
                train_label.append(val_label.iloc[val_idx])
                temp.append(val_idx)

    val_dataset.drop(temp, inplace=True, axis=0)
    val_label.drop(temp, inplace=True, axis=0)

    train_label_list = train_label['label'].values.tolist()
    val_label_list = val_label['label'].values.tolist()
    return train_dataset, val_dataset, train_label_list, val_label_list

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences
