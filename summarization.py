import openai
import json
import numpy as np
import time
import pickle
from tqdm import tqdm


def generate_answer(answer_context):
    try:
        completion = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo-0301",
                  messages=answer_context,
                  n=1)
    except:
        print("retrying due to an error......")
        time.sleep(20)
        return generate_answer(answer_context)

    return completion

def summarize_message(agent_contexts):
    prefix_string = "Here are a list of opinions from different agents: "

    for agent in agent_contexts:
        agent_response = agent[-1]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Write a summary of the different opinions from each of the individual agent."
    agent_context = [{"role": "user", "content": prefix_string}]
    completion = generate_answer(agent_context)
    content = completion["choices"][0]["message"]["content"]

    return content

def construct_message(summary, question):

    prefix_string = "Here is a summary of responses from other agents: {}".format(summary)

    prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response.".format(question)
    # prefix_string = prefix_string + "\n\n Using these opinions, can you provide an updated answer? Make sure to state your answer at the end of the response.".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}

def parse_answer(sentence):
    parts = sentence.split(" ")
    # Sequentially parse for the last number in the sentence

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
    answer = parse_answer("My answer is the same as the other agents and AI language model: the result of 12+28*19+6 is 550.")

    agents = 3
    rounds = 2
    np.random.seed(0)

    evaluation_round = 100
    scores = []

    generated_description = {}

    for round in tqdm(range(evaluation_round)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)

        answer = a + b * c + d - e * f
        agent_contexts = [[{"role": "user", "content": """What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.""".format(a, b, c, d, e, f)}] for agent in range(agents)]

        content = agent_contexts[0][0]['content']
        question_prompt = "We seek to find the result of {}+{}*{}+{}-{}*{}?".format(a, b, c, d, e, f)

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    # agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    # message = construct_message(agent_contexts_other, question_prompt)
                    summary = summarize_message(agent_contexts)
                    message = construct_message(summary, question_prompt)
                    agent_context.append(message)

                    print("message: ", message)
                completion = generate_answer(agent_context)
                # try:
                #     completion = openai.ChatCompletion.create(
                #               model="gpt-3.5-turbo-0301",
                #               messages=agent_context,
                #               n=1)
                # except:
                #     time.sleep(20)
                #     completion = openai.ChatCompletion.create(
                #               model="gpt-3.5-turbo-0301",
                #               messages=agent_context,
                #               n=1)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                print(completion)

        text_answers = []

        for agent_context in agent_contexts:
            text_answer = string =  agent_context[-1]['content']
            text_answer = text_answer.replace(",", ".")
            text_answer = parse_answer(text_answer)

            if text_answer is None:
                continue

            text_answers.append(text_answer)

            # print("text_answer: ", text_answer, answer)

            # if text_answer == answer:
            #     scores.append(1)
            # else:
            #     scores.append(0)

        generated_description[(a, b, c, d, e, f)] = (agent_contexts, answer)

        try:
            text_answer = most_frequent(text_answers)
            if text_answer == answer:
                scores.append(1)
            else:
                scores.append(0)
        except:
            continue

        print("performance:", np.mean(scores), np.std(scores) / (len(scores) ** 0.5))

    json.dump(generated_description, open("summarize_math_{}_{}.json".format(agents, rounds), "w"))
    # pickle.dump(generated_description, open("math_short_agents{}_rounds{}.p".format(agents, rounds), "wb"))
    import pdb
    pdb.set_trace()
    print(answer)
    print(agent_context)