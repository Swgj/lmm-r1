import requests
import json

data = {
    "query": [
        '<|im_start|>user\nIf A = {1,2,3} and B = {0,1,2}, find |A ∪ B|<|im_end|>\n<|im_start|>assistant\n<think>To find |A ∪ B|, I need to count all unique elements. A = {1,2,3} and B = {0,1,2}. The union is {0,1,2,3}, so |A ∪ B| = 4.</think><answer>The union A ∪ B = {0,1,2,3}, so |A ∪ B| = 4.</answer><|im_end|>'
    ],
    "prompts": [
        'If A = {1,2,3} and B = {0,1,2}, find |A ∪ B|'
    ]
}

response = requests.post("http://127.0.0.1:6000/get_reward", json=data)
print(json.dumps(response.json(), indent=2))