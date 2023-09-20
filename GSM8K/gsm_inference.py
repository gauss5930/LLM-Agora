import requests
import openai
import json
import numpy as np
import random
import time
from tqdm import tqdm
import argparse

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m1", "--model_1")
    parser.add_argument("-m2", "--model_2")
    parser.add_argument("-m3", "--model_3")
    parser.add_argument(
        "--API_KEY",
        type=str,
        help="your OpenAI API key to use gpt-3.5-turbo"
    )
    parser.add_argument(
        "--auth_token",
        type=str,
        help="your HuggingFace API key to use Inference Endpoints"
    )
    parser.add_argument("--round", default=2, type=int)
    parser.add_argument(
        "--cot",
        type=bool,
        help="If this is True, you can use Chain-of-Thought during inference."
    )
    parser.add_argument(
        "--output_dir",
        default="/Math",
        type=str,
        help="Directory to save the result file"
    )

    return parser.parse_args()

def load_json(prompt_path, endpoint_path):
    with open(prompt_path, "r") as prompt_file:
        prompt_dict = json.load(prompt_file)

    with open(endpoint_path, "r") as endpoint_file:
        endpoint_dict = json.load(endpoint_file)

    return prompt_dict, endpoint_dict

def construct_message(agents, instruction, idx):
    if len(agents) == 0:
        prompt = "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."
        return prompt
    
    contexts = [agents[0][idx]['content'], agents[1][idx]['content'], agents[2][idx]['content']]

    # system prompt & user prompt for gpt-3.5-turbo
    sys_prompt = f"I want you to act as a summarizer. You can look at multiple responses and summarize the main points of them so that the meaning is not lost. Multiple responses will be given, which are responses from several different models to a single question. And you should use your excellent summarizing skills to output the best summary."
    summarize_prompt = f"[Response 1]: {contexts[0]}\n[Response 2]: {contexts[1]}\nResponse 3: {contexts[2]}\n\nThese are response of each model to a certain question. Summarize comprehensively without compromising the meaning of each response."

    message = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": summarize_prompt},
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=message,
        max_tokens=256,
        n=1
    )

    prefix_string = f"This is the summarization of recent/updated opinions from other agents: {completion}"
    prefix_string = prefix_string + f"\n\n Use this summarization carefully as additional advice, can you provide your answer to the math problem? \n The original math problem is {instruction}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."
    return prefix_string

def generate_gsm(agents, question):
    agent_contexts = [[{"model": agent, "content": f"Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."}] for agent in agents]
    return agent_contexts

def read_jsonl(path: str):
    with open(path, "r") as fh:
        return [json.loads(line) for line in fh.readlines() if line]

if __name__ == "__main__":
    args = args_parse()
    openai.api_key = args.API_KEY
    model_list = [args.m1, args.m2, args.m3]

    prompt_dict, endpoint_dict = load_json("/src/prompt_template.json", "/src/inference_endpoint.json")

    def generate_answer(model, formatted_prompt):
        API_URL = endpoint_dict[model]
        headers = {"Authorization": f"Bearer {args.auth_token}"}
        payload = {"inputs": formatted_prompt}
        try:
            resp = requests.post(API_URL, json=payload, headers=headers)
            response = resp.json()
        except:
            print("retrying due to an error......")
            time.sleep(5)
            return generate_answer(API_URL, headers, payload)
        
        return {"model": model, "content": response[0]["generated_text"].split(prompt_dict[model]["response_split"])[-1]}
    
    def prompt_formatting(model, instruction, cot):
        if model == "alpaca" or model == "orca":
            prompt = prompt_dict[model]["prompt_no_input"]
        else:
            prompt = prompt_dict[model]["prompt"]
        
        if cot:
            instruction += "Let's think step by step."

        return {"model": model, "content": prompt.format(instruction)}

    agents = len(model_list)
    rounds = args.round
    random.seed(0)

    evaluation = 100

    generated_description = []

    questions = read_jsonl("data/GSM8K/gsm8k_test.jsonl")
    random.shuffle(questions)

    for idx in tqdm(range(evaluation)):
        question = questions[idx]["question"]
        answer = questions[idx]["answer"]

        agent_contexts = generate_gsm(model_list, question)

        print(f"# Question No.{idx} starts...")

        # Debate
        for debate in range(rounds+1):
            # Refer to the summarized previous response
            if debate != 0:
                message = construct_message(agent_contexts, question, 2 * debate - 1)
                agent_context.append(message)

            for agent_context in agent_contexts:
                # Generate new response based on summarized response
                completion = generate_answer(agent_context[-1]["model"], agent_context[-1]["content"] if debate != 0 else prompt_formatting(agent_context[-1]["model"], agent_context[-1]["content"], args.cot)["content"])
                agent_context.append(completion)

        print(f"# Question No.{idx} debate is ended.")

        models_response = {
            f"{args.m1}": [agent_contexts[0][1]["content"], agent_contexts[0][3]["content"], agent_contexts[0][-1]["content"]],
            f"{args.m2}": [agent_contexts[1][1]["content"], agent_contexts[1][3]["content"], agent_contexts[1][-1]["content"]],
            f"{args.m3}": [agent_contexts[2][1]["content"], agent_contexts[2][3]["content"], agent_contexts[2][-1]["content"]]
        }
        response_summarization = [
            agent_contexts[0][2], agent_contexts[0][4]
        ]
        generated_description.append({"question_id": idx, "question": question, "agent_response": models_response, "summarization": response_summarization, "answer": answer})

    if args.cot:
        file_name = "_cot.json"
    else:
        file_name = ".json"

    print(f"The result file 'gsm_result{file_name}' is saving...")
    with open(args.output_dir + f"/gsm_result{file_name}", "x") as f:
        json.dump(generated_description, f, indent=4)

    print("All done!!")