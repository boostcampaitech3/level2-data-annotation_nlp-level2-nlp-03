# text file
import json
from collections import defaultdict
import pandas as pd
import os
def return_dict():
    return {'idx':[], 'ent':[],'rel':[]}

def read_json(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    return json_data

def read_text(file_path):
    text = open(file_path,'r')
    total = ''
    line_idx = 1
    init_len = 0

    line_info = defaultdict(return_dict)
    line_content = []

    while True:
        line = text.readline()#.rstrip()
        if not line:
            break
        # print(line)
        line_content.append(line.rstrip())
        total += line
        line_info[line_idx]['idx'] = (init_len, init_len + len(line))
        line_idx += 1
        init_len = init_len + len(line)

    return line_info, line_content

def fill_ent(entities, line_infos):
    # 각 line에 쓰인 Entity 추가
    word_idx = entities['offsets'][0]['start']
    line_vals = {i:(line_infos[i]['idx'][0], line_infos[i]['idx'][1]) for i in range(1,len(line_infos)+1)}
    for idx, (s_idx, e_idx)  in line_vals.items():
        if s_idx<= word_idx < e_idx:
            line_infos[idx]['ent'].append(entities)
            break

def fill_relation(relations, line_info):
    # 각 line에 쓰인 relation 추가
    # 각 단어의 start num
    f_start, _ = return_idx(relations, 0)
    s_start, _ = return_idx(relations, 1)
    line_vals = {i: (line_info[i]['idx'][0], line_info[i]['idx'][1]) for i in range(1, len(line_info) + 1)}

    for idx, (s_idx, e_idx) in line_vals.items():
        if s_idx <= f_start < e_idx:
            if not(s_idx <= s_start < e_idx):
                continue
            line_info[idx]['rel'].append(relations)

                # breakpoint()
                # break
            # assert s_idx <= s_start < e_idx
            break

def return_idx(relations, rel_ent_idx):
    vals = relations['entities'][rel_ent_idx].split('|')[-1].split(',')
    return int(vals[0]), int(vals[1])

def update_dict(data_dict, sen_idx, line_content, subj_info, obj_info, rel_name,file_name):
    data_dict['sentence'].append(line_content[sen_idx-1])
    data_dict['subject_entity'].append({'word': subj_info[-2],
                                        'start_idx': subj_info[-1][0],
                                        'end_idx':subj_info[-1][1],
                                        'type':subj_info[1]})
    data_dict['object_entity'].append({'word': obj_info[-2],
                                        'start_idx': obj_info[-1][0],
                                        'end_idx':obj_info[-1][1],
                                        'type':obj_info[1]})
    data_dict['label'].append(rel_name)
    data_dict['file_name'].append(file_name)
    data_dict['sent_idx'].append(sen_idx-1)

### read data

with open('/opt/ml/test_data/BoostcampAI-NLP-03/annotations-legend.json') as f:
    relation_map = json.load(f)



def file_reads():
    import glob
    json = glob.glob("/opt/ml/test_data/BoostcampAI-NLP-03/ann.json/master/pool/*.json")
    html = glob.glob("/opt/ml/test_data/BoostcampAI-NLP-03/plain.html/pool/*.html")
    json.sort()
    html.sort()
    return json, html

base_json = "/opt/ml/test_data/BoostcampAI-NLP-03/ann.json/master/pool/"
base_text = "/opt/ml/test_data/raw_data/"

all_json, all_html = file_reads()

print(len(all_json), len(all_html))
# assert len(all_json) == len(all_html)
del_ent = ['e_90', 'e_92','e_141']
for itr, h_idx in enumerate(range(len(all_html))):
    whole_data = {
        'sentence': [],
        'subject_entity': [],
        'object_entity': [],
        'label': [],
        'file_name': [],
        'sent_idx': []
    }
    # if itr ==1:
    #     break
    # json_path = './test_data/earth.json'
    # txt_path = './test_data/earth.txt'
    print(all_html[h_idx])

    html_file = open(all_html[h_idx])
    html_content = html_file.read()
    start = html_content.find('data-origid=') + len('data-origid=')
    end = html_content.find('class=') - 1
    test_name = eval(html_content[start:end])
    txt_path = os.path.join(base_text, test_name)

    # all_html[h_idx].split('/')[-1].split('ann')[0]

    specific_json = all_html[h_idx].split('/')[-1].split('plain')[0]
    json_path = os.path.join(base_json, specific_json + 'ann.json')
    save_name = f"{test_name}.csv"

    print(test_name)
    print(json_path)

    # breakpoint()
    try:
        json_data = read_json(json_path)
    except:
        print('blocked')
        breakpoint()
        continue
    line_info, line_content = read_text(txt_path)
    file_name = (txt_path.split('/')[-1], json_path.split('/')[-1])


    for ent_num in range(len(json_data['entities'])):
        # 각 line에 쓰인 Entity 추가
        fill_ent(json_data['entities'][ent_num], line_info)


    for rel_idx in range(len(json_data['relations'])):
        # 각 line에 쓰인 relation 추가
        fill_relation(json_data['relations'][rel_idx], line_info)


    for line_k in line_info.keys():
        total_num_rel = len(line_info[line_k]['rel'])
        del_list = []
        for num_ent in range(total_num_rel):
            # relation 각각의 원소는 2개씩만 가지니까
            # 첫번째
            first_ent_idxs = line_info[line_k]['rel'][num_ent]['entities'][0].split('|')[-1].split(',')
            first_ent_leg = line_info[line_k]['rel'][num_ent]['entities'][0].split('|')[1]

            # 두번째
            second_ent_idxs = line_info[line_k]['rel'][num_ent]['entities'][1].split('|')[-1].split(',')
            second_ent_leg = line_info[line_k]['rel'][num_ent]['entities'][1].split('|')[1]

            f_idx, s_idx = int(first_ent_idxs[0]), int(second_ent_idxs[0])
            # line_content[1][f_f_idx:f_s_idx]
            do = True

            for e_idx, entity in enumerate(line_info[line_k]['ent']):
                if do:
                    if (entity['classId'], int(entity['offsets'][0]['start'])) == (first_ent_leg, f_idx):
                        if entity['classId'] in del_ent:
                            do = False
                            breakpoint()
                            continue

                        first_info = relation_map[entity['classId']].split('-')
                        first_info.append(entity['offsets'][0]['text'])
                        first_info.append((f_idx, f_idx + len(entity['offsets'][0]['text']) - 1))

                        del_list.append(e_idx)
                    elif (entity['classId'], int(entity['offsets'][0]['start'])) == (second_ent_leg, s_idx):
                        if entity['classId'] in del_ent:
                            breakpoint()
                            do = False
                            continue

                        second_info = relation_map[entity['classId']].split('-')
                        second_info.append(entity['offsets'][0]['text'])
                        second_info.append((s_idx, s_idx + len(entity['offsets'][0]['text']) - 1))

                        del_list.append(e_idx)
                    else:
                        pass
            if do:
                if first_info[0] == 'SUBJ':
                    subj_info = first_info
                    obj_info = second_info
                elif first_info[0] == 'OBJ':
                    subj_info = second_info
                    obj_info = first_info
                else:
                    raise NotImplementedError

                # relation 처리

                assert subj_info[2] == obj_info[2]
                rel_name = f'{subj_info[1].lower()}:{subj_info[2]}'#subj_info[2]

                update_dict(whole_data, line_k, line_content, subj_info, obj_info, rel_name,file_name)

        # 다하고나서 entity 길이에서 del_list 빼서 0이 아니면 걔네는 no_relation
        total_ent = [i for i in range(len(line_info[line_k]['ent']))]
        left_ent = list(set(total_ent) - set(del_list))

        if left_ent:

            # 저기 SUBJ, OBJ 처리 동일하게 하고
            if len(left_ent) !=2:
                print(line_k)
                continue

            # print('no_relation happen')
            # assert len(left_ent) ==2
            f = left_ent[0]
            s = left_ent[1]
            # relation 각각의 원소는 2개씩만 가지니까

            # 첫번째
            f_idx = line_info[line_k]['ent'][f]['offsets'][0]['start']
            first_ent_leg = line_info[line_k]['ent'][f]['classId']

            # 두번째
            s_idx = line_info[line_k]['ent'][s]['offsets'][0]['start']
            second_ent_leg = line_info[line_k]['ent'][s]['classId']

            do = True
            for e_idx, entity in enumerate(line_info[line_k]['ent']):
                if do:
                    if (entity['classId'], int(entity['offsets'][0]['start'])) == (first_ent_leg, f_idx):
                        if entity['classId'] in del_ent:
                            breakpoint()
                            do = False
                            continue
                        first_info = relation_map[entity['classId']].split('-')
                        first_info.append(entity['offsets'][0]['text'])
                        first_info.append((f_idx, f_idx + len(entity['offsets'][0]['text']) - 1))

                    elif (entity['classId'], int(entity['offsets'][0]['start'])) == (second_ent_leg, s_idx):
                        if entity['classId'] in del_ent:
                            breakpoint()
                            do = False
                            continue
                        second_info = relation_map[entity['classId']].split('-')
                        second_info.append(entity['offsets'][0]['text'])
                        second_info.append((s_idx, s_idx + len(entity['offsets'][0]['text']) - 1))

                    else:
                        pass
            if do:
                if first_info[0] == 'SUBJ':
                    subj_info = first_info
                    obj_info = second_info
                elif first_info[0] == 'OBJ':
                    subj_info = second_info
                    obj_info = first_info
                else:
                    raise NotImplementedError

                # relation 처리
                rel_name = 'no_relation'

                update_dict(whole_data, line_k, line_content, subj_info, obj_info, rel_name,file_name)


    ### done
    results = pd.DataFrame.from_dict(whole_data)
    new_name = save_name.split('.')[0]
    print(f'{new_name} done / len: ({len(whole_data["label"])})' )


    results.to_csv(os.path.join('/opt/ml/test_data/csv_data', new_name+'.csv'))





