import requests
import openai
import json
import numpy as np
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
        default="/Math/maht_result.json",
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
        prompt = "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."
        return prompt
    
    contexts = [agents[0][idx]['content'], agents[1][idx]['content'], agents[2][idx]['content']]

    sys_prompt = f"You are a helpful and precise assistant for summarizing the response of several models."
    summarize_prompt = f"[Response 1]: {contexts[0]}\n[Response 2]: {contexts[1]}\nResponse 3: {contexts[2]}\n\nThese are response of each model to a certain question. Summarize comprehensively without compromising the meaning of each response."

    # ChatGPT의 summarization 생성 코드 적어두기

    prefix_string = f"This is the summarization of recent/updated opinions from other agents: {}"
    prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response." + instruction
    return prefix_string

def generate_math(agents):
    a, b, c, d, e, f = np.random.randint(0, 30, size=6)

    answer = a + b * c + d - e * f
    question = "What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response."

    agent_contexts = [[{"model": agent, "content": question.format(a, b, c, d, e, f)}] for agent in agents]

    content = agent_contexts[0][0]["content"]

    question_prompt = f"We seek to find the result of {a}+{b}*{c}+{d}-{e}*{f}?"

    return agent_contexts, content, question_prompt, answer

def parse_answer(sentence):
    parts = sentence.split(" ")

    for part in parts[::-1]:
        try:
            answer = float(part)
            return answer
        except:
            continue

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

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
        
        return response[0]["generated_text"].split(prompt_dict[model]["response_split"])[-1]
    
    def prompt_formatting(model, instruction, cot):
        if model == "alpaca" or model == "orca":
            prompt = prompt_dict[model]["prompt_no_input"]
        else:
            prompt = prompt_dict[model]["prompt"]
        
        if cot:
            instruction += "Let's think step by step."

        return prompt.format(instruction)

    agents = len(model_list)
    rounds = 3
    np.random.seed(0)

    evaluation_round = 100
    scores = []

    generated_description = []

    for iteration in tqdm(range(evaluation_round)):
        agent_contexts, content, question_prompt, answer = generate_math(agents=model_list)

        # Debation
        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                # Refer to the summarized previous response
                if round != 0:
                    message = construct_message(agent_contexts, question_prompt, round)
                    agent_context.append(message)

                # Generate new response based on summarized response
                completion = generate_answer(agent_context["model"], prompt_formatting(agent_context))
                agent_context.append(completion)

        text_answers = []

        for agent in agent_contexts:
            text_answer = string = agent_context[-1]["content"]
            text_answer = text_answer.replace(",", ".")
            text_answer = parse_answer(text_answer)

            if text_answer is None:
                continue

            text_answers.append(text_answer)

        models_response = {
            f"{args.m1}": [agent_contexts[0][0]["content"], agent_contexts[0][1]["content"], agent_contexts[0][-1]["content"]],
            f"{args.m2}": [agent_contexts[1][0]["content"], agent_contexts[1][1]["content"], agent_contexts[1][-1]["content"]],
            f"{args.m3}": [agent_contexts[2][0]["content"], agent_contexts[2][1]["content"], agent_contexts[2][-1]["content"]]
        }
        generated_description.append({"question_id": iteration, "question": content, "agent_contexts": models_response, "answer": answer})

        try:
            text_answer = most_frequent(text_answers)
            if text_answer == answer:
                scores.append(1)
            else:
                scores.append(0)
        except:
            continue

    with open(args.output_dir, "x") as f:
        json.dump(generated_description, f, indent=4)