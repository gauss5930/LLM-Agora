from glob import glob
import pandas as pd
import json
import time
import random
import openai

# 모델에게 줄 메시지 생성. 원래 논문의 경우 ChatGPT를 사용했으므로 약간 message의 형태가 달라짐.
# 따라서 우리의 경우 각 모델의 prompt format을 따라서 작성하면 될 것 같음.
def construct_message(agents, question, idx):
    # 마지막 응답일 경우에는 agents는 빈 리스트가 되어 final answer을 출력하는 message를 줌.
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    # 다른 agent의 solution을 참고하도록 메시지를 줌.
    prefix_string = "These are the solutions to the problem from other agents: "

    # 각 agent의 response를 따로 꺼내서 "\n\n One agent solution: ```{}```".format(response)로 만듦.
    for agent in agents:
        agent_response = agent[idx]["content"]
        response = f"\n\n One agent solution: ```{agent_response}```"

        # 여러 agent의 응답을 하나로 다 합침.
        prefix_string = prefix_string + response

    # 모델의 응답들과 함께 이를 토대로 응답을 더욱 개선시켜서 줄 수 있도록 메시지를 줌. 이때 다른 모델의 응답과 question을 함께 줌.
    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
    # ChatGPT의 경우 {"role", "content"}의 형식을 지켜서 줘야 하지만, 우리의 경우 그냥 prompt만 줘도 될 것 같음. 
    return {"role": "user", "content": prefix_string}

# ChatGPT의 경우 출력만을 뽑아내기 위해서는 다음과 같이 여러 개의 인덱싱을 거쳐야 하는데, 우리는 그럴 필요는 없을 것 같음.
# 모델에 따라서 response_split으로 split만 진행하고 그 뒤의 응답만 가져오면 될 것 같음.
def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}

# ChatGPT API를 활용해서 출력 진행. 우리는 이것을 Inference Endpoint로 바꾸면 될 것 같음.
# Inference Endpoint에 오류가 발생한다면, 재시도할 수 있도록 generate_answer을 다시 시행.
# 서로 다른 모델로 출력을 진행해야 하니 answer_context 뿐만 아니라 모델 이름도 줘야할 것 같음.
# 그리고 Inference Endpoint는 ChatGPT API와 다를테니 이 점을 참고해서 inference code도 손봐야 할 것 같음.
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

# 객관식의 경우 다음과 같이 question & 문항을 분류함.
def parse_question_answer(df, ix):
    question = df.iloc[ix, 0]
    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]

    question = "Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {} Explain your answer, putting the answer in the form (X) at the end of your response.".format(question, a, b, c, d)

    answer = df.iloc[ix, 5]

    return question, answer

if __name__ == "__main__":
    # 자원의 제약 때문에 agent의 수는 3, round의 수는 2로 제한함.
    agents = 3
    rounds = 2

    # 디렉토리에 있는 csv파일을 모두 가져옴. -> List[str] 형태
    tasks = glob("/data/vision/billf/scratch/yilundu/llm_iterative_debate/mmlu/data/test/*.csv")

    # 여러개의 task csv 파일을 읽어서 하나의 데이터프레임으로 만듦.
    dfs = [pd.read_csv(task) for task in tasks]

    random.seed(0)
    response_dict = {}

    for i in range(100):
        # 여러 개의 task csv 파일을 묶어놓은 것 중에서 랜덤하게 하나를 선택함. 그리고 그 중에서 랜덤하게 하나를 선택함.
        df = random.choice(dfs)
        ix = len(df)
        idx = random.randint(0, ix-1)

        # question과 answer를 parsing함
        question, answer = parse_question_answer(df, idx)

        # agent의 수에 따라서 context를 여러 개 만듦.
        agent_contexts = [[{"role": "user", "content": question}] for agent in range(agents)]

        # round의 수만큼 반복함
        for round in range(rounds):
            # 
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    # 다른 agent의 response도 합쳐서 넣기
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2 * round - 1)
                    agent_context.append(message)

                # agent_context를 활용해서 answer 생성하기
                completion = generate_answer(agent_context)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                print(completion)

        response_dict[question] = (agent_contexts, answer)

    # model의 이름을 따와서 저장해야할 것 같음.
    json.dump(response_dict, open("mmlu_{}_{}.json".format(agents, rounds), "w"))