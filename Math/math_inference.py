import requests
import openai
import json
import numpy as np
import time
from tqdm import tqdm
import argparse

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_1", type=str)
    parser.add_argument("--model_2", type=str)
    parser.add_argument("--model_3", type=str)
    parser.add_argument(
        "--API_KEY",
        type=str,
        help="your OpenAI API key to use gpt-3.5-turbo"
    )
    parser.add_argument("--round", default=2, type=int)
    parser.add_argument(
        "--cot",
        default=False,
        action='store_true',
        help="If this is True, you can use Chain-of-Thought during inference."
    )
    parser.add_argument(
        "--output_dir",
        default="Math",
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

def construct_message(agent_context, instruction, idx):
    prefix_string = "Here are a list of opinions from different agents: "

    prefix_string = prefix_string + agent_context + "\n\n Write a summary of the different opinions from each of the individual agent."

    message = [{"role": "user", "content": prefix_string}]

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=message,
            max_tokens=256,
            n=1
        )['choices'][0]['message']['content']
    except:
        print("retrying ChatGPT due to an error......")
        time.sleep(5)
        return construct_message(agent_context, instruction, idx)

    prefix_string = f"Here is a summary of responses from other agents: {completion}"
    prefix_string = prefix_string + "\n\n Use this summarization carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response." + instruction
    return prefix_string

def summarize_message(agent_contexts, instruction, idx):
    prefix_string = "Here are a list of opinions from different agents: "

    for agent in agent_contexts:
        agent_response = agent[-1]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Write a summary of the different opinions from each of the individual agent."
    completion = construct_message(prefix_string, instruction, idx)

    return completion

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
    model_list = [args.model_1, args.model_2, args.model_3]

    prompt_dict, endpoint_dict = load_json("src/prompt_template.json", "src/inference_endpoint.json")

    def generate_answer(model, formatted_prompt):
        API_URL = endpoint_dict[model]["API_URL"]
        headers = endpoint_dict[model]["headers"]
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": 256
            }
        }
        try:
            resp = requests.post(API_URL, json=payload, headers=headers)
            response = resp.json()
        except:
            print("retrying due to an error......")
            time.sleep(5)
            return generate_answer(model, formatted_prompt)
        
        return {"model": model, "content": response[0]["generated_text"]}
    
    def prompt_formatting(model, instruction, cot):
        if model == "alpaca" or model == "orca":
            prompt = prompt_dict[model]["prompt_no_input"]
        else:
            prompt = prompt_dict[model]["prompt"]
        
        if cot:
            instruction += "Let's think step by step."

        return {"model": model, "content": prompt.format(instruction=instruction)}

    agents = len(model_list)
    rounds = args.round
    np.random.seed(0)

    evaluation = 100
    scores = []

    generated_description = []

    for round in tqdm(range(evaluation)):
        agent_contexts, content, question_prompt, answer = generate_math(agents=model_list)

        print(f"# Question No.{round+1} starts...")

        message = []

        # Debate
        for debate in range(rounds+1):
            # Refer to the summarized previous response
            if debate != 0:
                message.append(summarize_message(agent_contexts, question_prompt, 2 * debate - 1))
                for i in range(len(agent_contexts)):
                    agent_contexts[i].append(prompt_formatting(agent_contexts[i][-1]["model"], message[-1], args.cot))

            # Generate new response based on summarized response
            for agent_context in agent_contexts:
                completion = generate_answer(agent_context[-1]["model"], agent_context[-1]["content"])
                agent_context.append(completion)

        print(f"# Question No.{round+1} debate is ended.")

        text_answers = []

        for agent in agent_contexts:
            text_answer = string = agent[-1]["content"]
            text_answer = text_answer.replace(",", ".")
            text_answer = parse_answer(text_answer)

            if not text_answer:
                continue

            text_answers.append(text_answer)

        models_response = {
            f"{args.model_1}": [agent_contexts[0][1]["content"], agent_contexts[0][3]["content"], agent_contexts[0][-1]["content"]],
            f"{args.model_2}": [agent_contexts[1][1]["content"], agent_contexts[1][3]["content"], agent_contexts[1][-1]["content"]],
            f"{args.model_3}": [agent_contexts[2][1]["content"], agent_contexts[2][3]["content"], agent_contexts[2][-1]["content"]]
        }
        response_summarization = [
            message[0], message[1]
        ]
        generated_description.append({"question_id": round, "question": content, "agent_response": models_response, "summarization": response_summarization, "answer": str(answer)})

        try:
            text_answer = most_frequent(text_answers)
            if text_answer == answer:
                scores.append(1)
            else:
                scores.append(0)
        except:
            continue

    performance = {"performance": np.mean(scores)}
    print(f"The performance of {args.model_1} & {args.model_2} & {args.model_3}: ", performance["performance"])

    if args.cot:
        file_name = "_cot.json"
    else:
        file_name = ".json"

    print(f"The result file 'math_result{file_name}' is saving...")
    with open(args.output_dir + f"/math_result{file_name}", "x") as f:
        json.dump(generated_description, f, indent=4)

    print(f"The performance file 'math_performance{file_name}' is saving...")
    with open(args.output_dir + f"/math_performance{file_name}", "x") as f:
        json.dump(performance, f, indent=4)

    print("All done!!")