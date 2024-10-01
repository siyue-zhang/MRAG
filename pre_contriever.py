import pandas as pd
import json

file_path = './TempRAGEval/TempRAGEval.csv'
df = pd.read_csv(file_path).fillna('')
to_save = []

for index, row in df.iterrows():

    gold_evidences = []
    if row['gold_evidence_1']!='':
        gold_evidences.append(row['gold_evidence_1'])
    if row['gold_evidence_2']!='':
        gold_evidences.append(row['gold_evidence_2'])
    if row['time_relation']!='':
        # if row['id'] not in ['t_107','t_151','t_219','t_234','t_294','t_300']:
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
    to_save.append(example)

file_path = './TempRAGEval/TempRAGEval.json'
with open(file_path, 'w') as json_file:
    json.dump(to_save, json_file, indent=4, ensure_ascii=False)