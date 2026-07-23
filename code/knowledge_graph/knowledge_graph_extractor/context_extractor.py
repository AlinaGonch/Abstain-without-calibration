import json
from datasets import load_dataset


def extract_context(input_path):
    dataset = load_dataset('json', data_files=input_path)
    context_list = [item['context_'+str(i)] for i in range(1,6363) for item in dataset['train']]
    return context_list

if __name__ == "__main__":
    input_path = 'unique_contexts.json'
    kg_triplets = load_dataset('json',data_files='claude_kg_context.jsonl')['train']['result']['message']['content']
    kg_triplets = [kg_triplets[i][0]['text'] for i in range(0,6362)]
    context_list = extract_context(input_path)
    with open('extractor_train_data.jsonl', 'w', encoding='utf-8') as filename:
        for context, kg in zip(context_list, kg_triplets):
            filename.write(json.dumps({'context': context, 'kg_triplets': kg}) + '\n')
