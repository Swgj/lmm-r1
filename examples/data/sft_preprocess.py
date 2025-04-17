from datasets import load_dataset, Features, Value



data_name = 'a-m-team/AM-DeepSeek-R1-Distilled-1.4M'
# split = 'am_0.9M'
split = 'am_0.5M'
output_file = './data/'+split+'.jsonl'

features = Features({
    "messages": [
        {
            "role": Value("string"),
            "content": Value("string"),
            "info": {
                "source": Value("string"),
                "reference_answer": Value("string"),
                "test_case": Value("string"),
                "think_content": Value("string"),
                "answer_content": Value("string")
            }
        }
    ]
})

data = load_dataset(data_name, split, features=features, split='train')

def process_conversation(item):
    question = item['messages'][0]['content']
    response = item['messages'][1]['content']

    return {
        "question": question,
        "response": response,
    }

data = data.map(process_conversation, remove_columns=['messages'], num_proc=4)
data.to_json(output_file)