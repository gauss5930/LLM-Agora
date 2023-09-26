import json
import openai
import numpy as np
import time
import re
import argparse

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_1",
        type=str,
        help="It should be the same model used in gsm_inference.py"
    )
    parser.add_argument(
        "--model_2",
        type=str,
        help="It should be the same model used in gsm_inference.py"
    )
    parser.add_argument(
        "--model_3",
        type=str,
        help="It should be the same model used in gsm_inference.py"
    )
    parser.add_argument(
        "--cot",
        action="store_true"
    )
    parser.add_argument(
        "--output_dir",
        default="GSM8K",
        type=str
    )

    return parser.parse_args()

def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return float(matches[-1])
    
    return None

def parse_answer(input_str):
    pattern = r"([0-9]*)"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    if solution:
        return float(solution)
    else:
        return solution

def answer_check(List, answer):
    if answer in List: 
        return 1.0
    else:
        return 0.0

def compute_accuracy(gt, pred_solutions):
    answers = solve_math_problems(gt)

    if not answers:
        return None
    
    if type(pred_solutions) == list:
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer(pred_solution)

            if not pred_answer:
                pred_answer = solve_math_problems(pred_solution)

            pred_answers.append(pred_answer)

        return answer_check(pred_answers, answers)


if __name__ == "__main__":
    args = args_parse()
    model_list = [args.model_1, args.model_2, args.model_3]

    if args.cot:
        file_name = "_cot.json"
    else:
        file_name = ".json"

    with open(f"GSM8K/gsm_result{file_name}", "r") as f:
        response_dict = json.load(f)

    questions = [response_dict[i]["question"] for i in range(len(response_dict))]

    performance = []

    for turn in range(3):
        accuracies = []
        for idx in range(len(questions)):
            responses = [response_dict[idx]["agent_response"][model][turn] for model in model_list]
            gt = response_dict[idx]["answer"]

            accurate = compute_accuracy(gt, responses)

            if accurate is not None:
                accuracies.append(float(accurate))
            else:
                accuracies.append(0.0)

        performance.append({f"{turn}_performance": np.mean(accuracies)})
        print(performance)

    print(f"The performance file 'gsm_performance{file_name}' is saving...")
    with open(args.output_dir + f"/gsm_performance{file_name}", "x") as f:
        json.dump(performance, f, indent=4)

    print("All done!!")
