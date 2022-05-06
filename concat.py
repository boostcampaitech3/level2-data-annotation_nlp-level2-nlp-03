import pandas as pd
import glob

all_csv = glob.glob("/opt/ml/test_data/csv_data/*.csv")
pd_list = []
for csv_path in all_csv:
    tt = pd.read_csv(csv_path).drop(columns='Unnamed: 0')
    # new = tt.drop(columns='Unnamed: 0')
    pd_list.append(tt)
concat = pd.concat(pd_list,ignore_index=True)
concat['id'] = concat.index

concat.to_csv('/opt/ml/test_data/nlp3_all_data_new.csv')

labels =[
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
print(len(labels))
import pickle

label_to_num = {v:idx for idx,v in enumerate(labels)}
num_to_label = {idx:v for idx,v in enumerate(labels)}

with open('nlp3_dict_label_to_num.pkl' ,'wb') as f:
    pickle.dump(label_to_num, f)

with open('nlp3_dict_num_to_label.pkl' ,'wb') as f:
    pickle.dump(num_to_label, f)
