import json
import os
import random
import re
from argparse import ArgumentParser
from concurrent import futures

import Levenshtein
from flask import Flask, jsonify, request
import requests  # 添加用于API调用
import time  # 添加用于API限速

from loguru import logger

app = Flask(__name__)

problem_to_answer = {}

# LLM API配置
LLM_API_CONFIG = {
    "api_key": "sk-9c79f2bc5e8b4398b5e451d3fd016ccc",  # 需要设置API密钥
    "endpoint": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "temperature": 0.0,
    "max_tokens": 50,
    "retry_limit": 3,
    "retry_delay": 5,
    "timeout": 30
}


def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


def verify_format(content):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count = content.count("<think>")
    answer_count = content.count("<answer>")
    return bool(re.match(format_pattern, content, re.DOTALL)) and think_count == 1 and answer_count == 1


def find_similar_problem(problem):
    max_sim = -1
    target_problem = None
    for p in problem_to_answer.keys():
        sim = Levenshtein.ratio(problem, p)
        if sim > max_sim:
            max_sim = sim
            target_problem = p
    return target_problem


def extract_content(text):
    """提取<think>和<answer>标签中的内容"""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    
    think_content = think_match.group(1).strip() if think_match else ""
    answer_content = answer_match.group(1).strip() if answer_match else ""
    
    return think_content, answer_content


def verify_llm(content, reference_answer):
    """使用LLM API评估答案的准确性"""
    try:
        # 提取用户的思考过程和最终答案
        _, answer_content = extract_content(content)
        if not answer_content:
            logger.error("Could not extract answer content")
            return 0.0
            
        # 构建评估提示
        prompt = f"""
You are an expert for multi-domain reasoning tasks. Analyze responses based on these universal criteria:
{
  "criteria": {
    "correctness": {
      "math": "Symbolic equivalence via SymPy (e.g., x² vs x*x)",
      "text": "Core concept overlap ≥80% + logical entailment",
    },
    "conciseness": {
      "math": "Step count ≤120% of reference",
      "text": "Length ≤150% of reference + BERTScore ≥0.8"
    }
  },
  "scoring": {
    "3": "Correct & optimal conciseness",
    "2": "Correct but redundant",
    "0": "Incorrect"
  }
}
reference answer: {reference_answer}
user answer: {answer_content}
Your scoring(only a number in 0,2,3 is allowed):
"""

        # 调用LLM API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_CONFIG['api_key']}"
        }
        
        data = {
            "model": LLM_API_CONFIG["model"],
            "messages": [
                {"role": "system", "content": "You serve as an impartial scoring assistant. It is your task to evaluate whether the user's answer is mathematically equivalent to the reference answer. Respond with a single numerical score."},
                {"role": "user", "content": prompt}
            ],
            "temperature": LLM_API_CONFIG["temperature"],
            "max_tokens": LLM_API_CONFIG["max_tokens"]
        }
        
        # 添加重试机制
        for attempt in range(LLM_API_CONFIG["retry_limit"]):
            try:
                response = requests.post(
                    LLM_API_CONFIG["endpoint"], 
                    headers=headers, 
                    json=data,
                    timeout=LLM_API_CONFIG["timeout"]
                )
                
                if response.status_code == 200:
                    result = response.json()
                    score_text = result["choices"][0]["message"]["content"].strip()
                    
                    # 提取分数 (尝试匹配浮点数)
                    score_match = re.search(r"(\d+\.\d+|\d+)", score_text)
                    if score_match:
                        score = float(score_match.group(1))
                        # 确保分数在0-1范围内
                        return max(0.0, min(1.0, score))
                    else:
                        logger.error(f"Could not extract score from LLM response: {score_text}")
                        return 0.0
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    if attempt < LLM_API_CONFIG["retry_limit"] - 1:
                        time.sleep(LLM_API_CONFIG["retry_delay"])
                    else:
                        return 0.0
            except Exception as e:
                logger.error(f"Error calling LLM API: {str(e)}")
                if attempt < LLM_API_CONFIG["retry_limit"] - 1:
                    time.sleep(LLM_API_CONFIG["retry_delay"])
                else:
                    return 0.0
    except Exception as e:
        logger.error(f"Error in verify_llm: {str(e)}")
        return 0.0


@app.route("/get_reward", methods=["POST"])
def get_reward():
    # 获取请求中的 JSON 数据
    data = request.get_json()
    # 检查是否有 'query' 字段
    if "query" not in data:
        return jsonify({"error": "queries field is required"}), 400
    rewards = []
    format_rewards = []
    acc_rewards_futures = []
    for q, problem in zip(data["query"], data["prompts"]):
        if problem is None:
            return jsonify({"error": f"problem not found from {q}"}), 400
        if problem not in problem_to_answer:
            # This should not happen
            print(f"problem not exists: {problem}")
            problem = find_similar_problem(problem)
        answer = problem_to_answer[problem]
        response = get_response_from_query(q) or q
        if response is None:
            return jsonify({"error": f"response not found from {q}"}), 400
        format_reward = float(verify_format(response)) * 0.5
        acc_reward_future = llm_verify_executor.submit(verify_llm, response, answer)
       
        do_print = random.randint(1, 20) == 1
        if do_print:
            info = f"Query: {q}\n\nProblem: {problem}\n\n Answer: {answer}\n\n Response: {response}\n\n Format Reward: {format_reward}\n\n"
            info = re.sub(r"<\|.*?\|>", "", info)
            logger.info(info)
            
        format_rewards.append(format_reward)
        acc_rewards_futures.append(acc_reward_future)
    acc_rewards = [f.result() for f in acc_rewards_futures]
    rewards = [f + a for f, a in zip(format_rewards, acc_rewards)]
    # 返回包含 rewards 的响应
    return jsonify({"rewards": rewards, "format_rewards": format_rewards, "acc_rewards": acc_rewards})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default=None, help="Datasets to use (comma separated)", required=True
    )
    parser.add_argument(
        "--prompt-template", type=str, default=None, help="Prompt template", required=True
    )
    parser.add_argument(
        "--input_key", type=str, default="prompt", help="The key name of prompt."
    )
    parser.add_argument(
        "--log_file", type=str, default="remote_rm.log", help="Log file path"
    )
    parser.add_argument(
        "--api_key", type=str, default="", help="LLM API key"
    )
    parser.add_argument(
        "--api_endpoint", type=str, default="https://api.deepseek.com/chat", 
        help="LLM API endpoint"
    )
    parser.add_argument(
        "--api_model", type=str, default="deepseek-chat", help="LLM model to use"
    )
    args = parser.parse_args()
    
    # 更新API配置
    LLM_API_CONFIG["api_key"] = args.api_key
    LLM_API_CONFIG["endpoint"] = args.api_endpoint
    LLM_API_CONFIG["model"] = args.api_model
    
    if os.path.exists(args.log_file):
        os.remove(args.log_file)
    logger.remove()
    logger.add(args.log_file)
    # Split dataset paths and load all datasets
    dataset = []
    for dataset_path in args.dataset.split(','):
        dataset_path = dataset_path.strip()
        if dataset_path.endswith("json"):
            with open(dataset_path, "r") as f:
                dataset.extend(json.load(f))
        elif dataset_path.endswith("jsonl"):
            with open(dataset_path, "r") as f:
                dataset.extend([json.loads(l) for l in f.readlines()])
        else:
            raise ValueError(f"Unsupported file format for dataset: {dataset_path}")

    format_pattern = r"^<think>(?:(?!</think>).)*</think><answer>(?:(?!</answer>).)*</answer>\Z"

    if args.prompt_template=="chatml":
        problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
        response_prefix = r"<\|im_start\|>assistant\n"
    elif args.prompt_template=="qwen1":
        problem_pattern = r"｜User｜>(.*?)<｜Assistant｜>"
        response_prefix = r"<｜Assistant｜>"
    elif args.prompt_template=="base":
        problem_pattern = r"User: (.*?)\n\nAssistant:"
        response_prefix = r"Assistant: "
    elif args.prompt_template=="phi3":
        problem_pattern = r"<|user|>\n(.*?)<|end|>\n<|assistant|>\n"
        response_prefix = r"<|assistant|>\n"
    elif args.prompt_template=="phi4":
        problem_pattern = r"<|user|>\n(.*?)<|end|>\n<|assistant|>\n"
        response_prefix = r"<|assistant|>\n"
    else:
        raise ValueError(f"Unknown chat format: {args.prompt_template}")
        
    print("load dataset success")
    for item in dataset:
        problem = item[args.input_key]
        answer = item["answer"].strip()
        problem_to_answer[problem] = answer

    # 使用进程池来处理LLM API请求
    llm_verify_executor = futures.ProcessPoolExecutor(max_workers=8)

    app.run(host="0.0.0.0", port=6000, debug=False, use_reloader=False)
    llm_verify_executor.shutdown()