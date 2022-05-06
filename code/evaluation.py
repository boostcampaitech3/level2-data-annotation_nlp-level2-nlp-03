import pickle as pickle
import os
import json
import pandas as pd
import argparse
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score

def klue_re_micro_f1(preds, labels, average):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def label_to_num(label):
  num_label = []

  dict_label_to_num = {'no_relation': 0, 'org:top_members/employees': 1, 'org:members': 2, 'org:product': 3, 'per:title': 4, 'org:alternate_names': 5, 'per:employee_of': 6, 'org:place_of_headquarters': 7, 'per:product': 8, 'org:number_of_employees/members': 9, 'per:children': 10, 'per:place_of_residence': 11, 'per:alternate_names': 12, 'per:other_family': 13, 'per:colleagues': 14, 'per:origin': 15, 'per:siblings': 16, 'per:spouse': 17, 'org:founded': 18, 'org:political/religious_affiliation': 19, 'org:member_of': 20, 'per:parents': 21, 'org:dissolved': 22, 'per:schools_attended': 23, 'per:date_of_death': 24, 'per:date_of_birth': 25, 'per:place_of_birth': 26, 'per:place_of_death': 27, 'org:founded_by': 28, 'per:religion': 29}
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def make_probs(probs):
  prob_temp = []
  probs = probs.apply(lambda x: x[1:-1].split(','))
  for i in probs:
    prob_temp.append(list(map(float, i)))
  return prob_temp

def evaluation(gt_path, pred_path):

  pred = pd.read_csv(pred_path) # model이 예측한 정답 csv [id,pred_label, probs]
  answer = pd.read_csv(gt_path) # test dataset의 정답 csv [id, label]

  pre_pred = pred.loc[pred['id'][answer['id']]]

  micro_f1 = klue_re_micro_f1(label_to_num(pre_pred["pred_label"]), label_to_num(answer["label"]), average='micro') 
  auprc = klue_re_auprc(np.array(make_probs(pre_pred['probs'])), np.array(label_to_num(answer["label"])))

  results = {}
  results['micro_f1'] = {
        'value': f'{micro_f1:.04f}',
        'rank': True,
        'decs': True,
    }
  results['auprc'] = {
      'value': f'{auprc:.04f}',
      'rank': False,
      'decs': True,
  }

  return json.dumps(results)

def main(args):
  print(evaluation(args.public_dataset_dir, args.pred_answer_dir))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # data dir
  parser.add_argument('--pred_answer_dir', type=str, default="./submission.csv")
  parser.add_argument('--public_dataset_dir', type=str, default="./ground_truth/private_gt.csv")
  args = parser.parse_args()
  main(args)



