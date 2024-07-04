#!/usr/bin/env python
import records
import json
from tqdm import tqdm
import re

def execute_raw_query(fdb, query):
    db = records.Database('sqlite:///{}'.format(fdb))
    try:
        out = db.query(query)
        return [dict(row) for row in out]
    except Exception as e:
        return repr(e)

def remove_extra_whitespaces(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



def process_model(path_db, path_jsonl, path_log,splitter):
    with open(path_jsonl, 'r', encoding='utf8') as fp:
        data = [json.loads(line) for line in fp.readlines()]

    grades = []
    exact_match = []
    log_entries = []
    
    for entry in tqdm(data):
        gold_query = remove_extra_whitespaces(entry["output"])
        pred_query = remove_extra_whitespaces(entry["prediction"].split(splitter)[0].strip())

        gold_result = execute_raw_query(path_db, gold_query)
        pred_result = execute_raw_query(path_db, pred_query)
        
        correct = pred_result == gold_result
        match = gold_query == pred_query
        
        grades.append(correct)
        exact_match.append(match)
        
        log_entry = {
            "truth_query": gold_query,
            "prediction_query": pred_query,
            "truth_result": gold_result,
            "prediction_result": pred_result,
            "match": match,
            "exact_match": correct
        }
        log_entries.append(log_entry)

    with open(path_log, 'w', encoding="utf8") as log_file:
        for entry in log_entries:
            log_file.write(json.dumps(entry) + '\n')

    print(json.dumps({
        'ex_accuracy': sum(grades) / len(grades),
        'lf_accuracy': sum(exact_match) / len(exact_match),
    }, indent=2))

if __name__ == '__main__':
    path_db = "data.db"
    results=[
            ("llama2-base-predictions","</s>"),
            ("llama2-tr-predictions","</s>"),
            ("llama3-base-predictions","<|eot_id|>"),
            ("llama3-tr-predictions","<|eot_id|>"),
            ("phi3-base-predictions","\n"),
            ("phi3-tr-predictions","\n"),
            ]
    
    for result in results:
        model,splitter = result
        path_jsonl = f"{model}.jsonl"
        path_log = f"Results/{model}-goldpred.txt"
        print(model)
        process_model(path_db, path_jsonl, path_log,splitter)