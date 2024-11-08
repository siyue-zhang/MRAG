import pandas as pd
import json
from utils import remove_implicit_condition

file_path = './TempRAGEval/TempRAGEval.csv'
df = pd.read_csv(file_path).fillna('')
to_save = []
normalize = False

for index, row in df.iterrows():
    if row['id']=='':
        continue
    print(index)
    gold_evidences = []
    if row['gold_evidence_1']!='':
        gold_evidences.append(row['gold_evidence_1'])
    if row['gold_evidence_2']!='':
        gold_evidences.append(row['gold_evidence_2'])
    if row['time_relation']!='':
        # if row['id'] not in ['t_580']:
        assert row['time_relation'] in row['question'], row

    example = {
        'question': row['question'].strip().replace('\n',''),
        'answers': [item.strip() for item in row['answer'].split(' | ')],
        'exact': row['exact_time'],
        'source': row['original_dataset'],
        'gold_evidences': gold_evidences,
        'time_relation': row['time_relation'],
        'id': row['id']
    }

    if normalize:
        question = example['question']
        time_relation = example['time_relation']
        if time_relation != '':
            parts = question.split(time_relation)
            no_time_question = time_relation.join(parts[:-1])
        else:
            no_time_question = question
        normalized_question, implicit_condition = remove_implicit_condition(no_time_question)
        normalized_question = normalized_question[:-1] if normalized_question[-1] in '.?!' else normalized_question
        example['ori_question'] = question
        example['question'] = normalized_question

    to_save.append(example)

norm = '_norm' if normalize else ''
file_path = f'./TempRAGEval/TempRAGEval{norm}.json'
with open(file_path, 'w') as json_file:
    json.dump(to_save, json_file, indent=4, ensure_ascii=False)