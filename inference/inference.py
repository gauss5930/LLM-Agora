import requests
import json
import openai
import time
import random
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_1", type=str)
    parser.add_argument("--model_2", type=str)
    parser.add_argument("--model_3", type=str)
    parser.add_argument("--API_KEY", type=str)
    parser.add_argument("--auth_token", type=str)
    parser.add_argument("--round", default=2, type=int)
    parser.add_argument("--cot", default=False, type=bool)

    return parser.parse_args()

def load_json(prompt_path, endpoint_path):
    with open(prompt_path, "r") as prompt_file:
        prompt_dict = json.load(prompt_file)

    with open(endpoint_path, "r") as endpoint_file:
        endpoint_dict = json.load(endpoint_file)

    return prompt_dict, endpoint_dict

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return 
    
    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["response"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string += response

    prefix_string += "\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."
    
    return prefix_string

def prompt_formatting(instruction, input, prompt_dict, model, cot):
    if model == "alpaca" or model == "orca":
        if input:
            prompt = prompt_dict[model]["prompt_input"]
        else:
            prompt = prompt_dict[model]["prompt_no_input"]
    else:
        prompt = prompt_dict[model]["prompt"]
    
    if cot:
        instruction += "Let's think step by step."

    return prompt.format(instruction)

def generate_answer(API_URL, headers, payload):
    try:
        resp = requests.post(API_URL, json=payload, headers=headers)
        response = resp.json()
    except:
        print("retrying due to an error......")
        time.sleep(10)
        return generate_answer(API_URL, headers, payload)

    return response[0]["generated_text"]

def inference():
    args = arg_parse()

    prompt_dict, endpoint_dict = load_json("/prompt_template.json", "/inference_endpoint.json")

    model_list = [args.model_1, args.model_2, args.model_3]

    agents = len(model_list)
    rounds = args.round

    for i in range(100):


        for round in range(rounds):
            for model in model_list:
                formatted_prompt = prompt_formatting(instruction=instruction, input=input, prompt_dict=prompt_dict, model=model, cot=args.cot)
                API_URL = endpoint_dict[model]
                headers = {"Authorization": f"Bearer {args.auth_token}"}
                payload = {"inputs": formatted_prompt}





if __name__ == "__main__":
    inference()