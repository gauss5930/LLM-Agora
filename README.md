<p align="center"><img src="/assets/LLM-Agora.png", width='250', height='250'></p>

# LLM Agora
LLM Agora is the place to debate between open-source LLMs and revise their responses!

The **LLM Agora** 🗣️🏦 aims to improve the quality of open-source LMs' responses through debate & revision introduced in [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325).
We would like to thank the authors of the paper for their brilliant ideas that allowed me to pursue this project.

Do you know that? 🤔 **LLMs can also improve their responses by debating with other LLMs**! 😮 We tried to apply this concept to several open-source LMs to verify that the open-source model, not the proprietary one, can sufficiently improve the response through discussion. 🤗
For this, we developed **LLM Agora**!
You can try the LLM Agora and check the example responses in [LLM Agora Spaces](https://huggingface.co/spaces/Cartinoe5930/LLMAgora)!

We tried to follow the overall framework of [llm_multiagent_debate](https://github.com/composable-models/llm_multiagent_debate), and we added additional things such as CoT.
We could confirm that through the experiments of LLM Agora, although there are still shortcomings, open-source LLMs can also improve the quality of models' responses through multi-agent debate.

## ToC

1. [Introduction & Motivation](#introduction--motivation)
2. [What is LLM Agora?](#what-is-llm-agora)
3. [Experiments](#experiments)
4. [Analysis](#analysis)
5. [Future work](#future-work)
6. [How to do?](#how-to-do)

## Introduction & Motivation

The LLM Agora project is inspired by the multi-agent debate introduced in the paper '[Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325)' as mentioned above.
Therefore, before start introducing the LLM Agora, we would like to explain the concept of multiagent debate!

With the remarkable development of LLM, LLM has become capable of outputting responses at a significantly higher level.
For example, GPT-4 is enough to pass even difficult exams. 
Despite the brilliant performance of proprietary LLMs, their first responses have some errors or mistakes. 
Then, how can correct and revise the responses? 
In the paper, they suggested that debate between several agents can revise the responses and improve the performance!
Through several experiments, the fact that this method can correct the errors in responses and revise the quality of responses was proved. (If you want to know more, please check the official [GitHub Page of paper](https://composable-models.github.io/llm_debate/)!)

In the paper, the overall experiment is conducted using only one model, but in the Analysis part, it is said that a synergy effect that shows further improved performance can be seen when different types of LLM are used.
The LLM Agora is exactly inspired from this point! 

We started the LLM Agora project with the expectation that if several open-source LLMs create a synergy effect through debate between other models, we can expect an effect that can complement the shortcomings of open-source LLM, which still has some shortcomings.
Therefore, we carried out the LLM Agora project because we thought it could be a groundbreaking method if multi-agent debate could improve the quality of responses of open-source LLMs.

## What is LLM Agora?

The meaning of '[Agora](https://en.wikipedia.org/wiki/Agora)' is a place where meetings were held in ancient Greece.
We thought this meaning was similar to a multi-agent debate, so we named it **LLM Agora**.
The summarized difference between multi-agent debate and LLM Agora is as follows:

1. **Models**: **Several open-source LLMs** were utilized, unlike the paper that used proprietary LLM(ChatGPT).
In addition, we analyzed whether using open-source LLM in multi-agent debate is effective or not, and used various models to check the synergy effect.
2. **Summarization**: The concatenated response was used for the debate sentence in the paper. However, according to the experimental result of the paper, it is more effective to summarize the models' responses and use them as a debate sentence. Therefore, we summarized the models' responses with ChatGPT and used it as a debate sentence.
3. **Chain-of-Thought**: We used **Chain-of-Thought** in a multi-agent debate to confirm whether open-source LLM can achieve performance improvement through Chain-of-Thought and to determine its impact on the debate.
4. **HuggingFace Space**: We implemented LLM Agora in HuggingFace Space so that people can directly use LLM Agora and check the responses generated through experiments.
It's open to everyone, so check it out! [LLM Agora Space](https://huggingface.co/spaces/Cartinoe5930/LLMAgora)

We hope that LLM Agora will be used in the future as a way to improve the performance of open-source models as well as proprietary models. 
Once again, we would like to thank the authors of the '[Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325)' for suggesting the idea of multiagent-debate.

## Experiments

We followed the experiments progressed in the paper to prove the effectiveness of multi-agent debate on various open-source LLMs.
The goal of experiments is as follows:

- Effects of using open-source models for multi-agent debate
- Impact of CoT on open-source models and multi-agent debate
- Synergies of using diverse models

### Experimental setup

#### Tasks

We experimented using the same task in the paper.
The tasks on which the experiment was performed are as follows: 

- **Math**: The problem of arithmetic operations on six randomly selected numbers. The format is `{}+{}*{}+{}-{}*{}=?`
- **GSM8K**: GSM8K is a dataset consisting of high-quality linguistically diverse grade school math word problems.
- **MMLU**: MMLU is a benchmark covering 57 subjects across STEM, the humanities, the social sciences, and more.

For all tasks, only 100 questions were sampled and used in the experiment.

#### The number of agents & rounds

The multi-agent debate has some special parameters such as the number of **agents** and **rounds**.
Each means **the number of used models for debate** and **the number of will be conducted debate rounds**.
The number of agents and rounds were set to **3** and **2**, respectively, due to the resource issue.

#### Baselines & Summarizer model

The models were deployed with [HuggingFace Inference Endpoints](https://huggingface.co/inference-endpoints), but for some reason, models based on LLaMA1 cannot be deployed with Inference Endpoints, so models based on LLaMA2 were mainly used.
In addition, GPTQ models were used to reduce the model size for deployment as an Inference Endpoint.
Thank you to [TheBloke](https://huggingface.co/TheBloke) for uploading the GPTQ model.
The models in **bold** are the baseline models used in the LLM Agora experiment.

- **Llama2-13B**: https://huggingface.co/TheBloke/Carl-Llama-2-13B-GPTQ
- Llama2-13B-Chat: https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ
- Vicuna2-13B: https://huggingface.co/TheBloke/vicuna-13B-v1.5-GPTQ
- **WizardLM2-13B**: https://huggingface.co/TheBloke/WizardLM-13B-V1.2-GPTQ
- **Orca2-13B**: https://huggingface.co/TheBloke/OpenOrcaxOpenChat-Preview2-13B-GPTQ
- Falcon-7B: https://huggingface.co/tiiuae/falcon-7b
- Falcon-7B-Instruct: https://huggingface.co/tiiuae/falcon-7b-instruct

We also used ChatGPT(gpt-3.5-turbo) as a summarizer model that summarizes the models' responses.

#### Prompt Format

Please check the `src/prompt_template.json`!

### Results: Math

Math task is the problem of finding the final value when four arithmetic operations are performed on six randomly given numbers.
(ex. `What is the result of 13+17*4+30-12*22?`)
Although it showed slightly worse performance due to poor math skills, which is a weakness in LLM, performance improved through the debate process both when CoT was used and it was not used.
In addition, the effect of CoT is not much at the beginning of the debate, but after the final debate, the positive effect of CoT could be confirmed.
You can check the responses to Math of each model in [LLM Agora Space](https://huggingface.co/spaces/Cartinoe5930/LLMAgora) or `Math/math_result.json` and `Math/math_result_cot.json`.
The result of the Math task is as follows:

**Math Result**

|Response|None|CoT|
|---|---|---|
|1st response|5|5|
|2nd response|11|11|
|3rd response|10|**17**|

<img src="https://github.com/gauss5930/LLM-Agora/blob/main/assets/Math_performance.png">

### Results: GSM8K

GSM8K is the high-quality linguistically diverse grade school math word problem.
The result of GSM8K showed that, unlike the result of other tasks, the models did not gain much benefit from debate and CoT.
There was no change in performance despite the debate when CoT was not used, and performance got worse as the debate progressed when CoT was used.
You can check the responses to GSM8K of each model in [LLM Agora Space](https://huggingface.co/spaces/Cartinoe5930/LLMAgora) or `GSM8K/gsm_result.json` and `GSM8K/gsm_result_cot.json`.
The result of GSM8K is as follows:

**GSM8K Results**

|Response|None|CoT|
|---|---|---|
|1st response|26|**28**|
|2nd response|26|26|
|3rd response|26|23|

<img src="https://github.com/gauss5930/LLM-Agora/blob/main/assets/GSM8K_performance.png">

### Results: MMLU

MMLU is a benchmark that covers 57 subjects across STEM, the humanities, the social sciences, and more.
The result of MMLU showed that performance is improving through CoT and debate, but strangely, the performance plummets to 0 after the first debate.
You can check the responses to the MMLU of each model in [LLM Agora Space](https://huggingface.co/spaces/Cartinoe5930/LLMAgora) or `MMLU/mmlu_result.json` and `MMLU/mmlu_result_cot.json`.
The result of MMLU is as follows:

**MMLU Results**

|Response|None|CoT|
|---|---|---|
|1st response|48|50|
|2nd response|0|0|
|3rd response|54|**58**|

<img src="https://github.com/gauss5930/LLM-Agora/blob/main/assets/MMLU_performance.png">

## Analysis

The experiments of LLM Agora are performed on Math, GSM8K, and MMLU.
The results showed that although open-source LLMs have some shortcomings in performing the multi-agent debate method, it is also effective in open-source LLMs. 
In addition, we were able to confirm that performance improved even when CoT was used.
However, the performance improvement was not significant, and in the case of GSM8K, it was not affected by debate & CoT.

In addition, the quality of the responses to each task of the models was not good.
Since the quality of the responses is not good, it seems that the multi-agent debate hurt the quality of the responses.
The analysis of the experiment results is summarized as follows:

- Open-source LLMs can benefit from multi-agent debate and CoT when models output proper quality responses.
- We did not investigate the synergy effect that occurs when using various models in the experiment. However judging from the results of deteriorating performance through debate, it may be possible to demonstrate a good synergy effect if the models' responses are high-quality, but if this is not the case it was confirmed that it could worsen performance.

Although the LLM Agora that utilized open-source LLMs has some shortcomings, we confirmed that multi-agent debate can improve the performance of models.
Therefore, multi-agent debate could become an effective method to improve the quality of models' responses if additional improvements are made to the open-source models and multi-agent debate.
We hope that LLM Agora will help research methods to improve the performance of open-source models, and appreciate the authors of '[Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325)' for suggesting multi-agent debate, which is the motivation of LLM Agora.

## Future work

As we mentioned in 'Analysis', there were some obstacles to performing the multi-agent debate with open-source LLMs.
Therefore, we will try to utilize more improved open-source LLMs or research the method that can improve the quality of models' responses to make the multi-agent debate sufficiently effective to improve the performance of open-source LLMs.
In addition, since the resource issue, LLM Agora supports just 7 models, however, we will try to develop the LLM Agora can support more various open-source LLMs!

## How to do?

The following description is the process of our experiment. Please follow the process of our experiment, if you want to conduct them!
We would like to note that we don't provide the inference endpoint APIs. 
Therefore, we recommend creating your inference endpoint API if you want to conduct the experiments.

0. [**Setup inference endpoint**](#setup-inference-endpoint)
1. [**Requirements**](#requirements)
2. [**Inference**](#inference)
3. [**Evaluation**](#evaluation)

### Setup inference endpoint

As we mentioned above, we don't provide any inference endpoint API.
Therefore, you should create your inference endpoint API if you want to conduct the experiments.
The process of setup inference endpoint is as follows:

1. Create your inference endpoint API using [HuggingFace Inference Endpoints](https://huggingface.co/inference-endpoints) for the models mentioned in the **Experimental setup**.
2. Fill in the blanks of `src/inference_endpoint.json` with your inference endpoint API. `src/inference_endpoint.json` will be used when performing inference. 

### Requirements

To install the required library, just run these two lines of code!

```
%cd LLM-Agora
pip install -r src/requirements.txt 
```

### Inference

You can do inference by executing the following Math, GSM8K, and MMLU codes. 
At this time, you can do inferences using CoT by adding just one line, `--cot`.
In addition, by executing the 'Custom Inference' code, inference about custom instructions is possible.

**Math**
```
python Math/math_inference.py \
    --model_1 llama \
    --model_2 wizardlm \
    --model_3 orca \
    --API_KEY your_OpenAI_API_KEY \
    --cot    # If you write '--cot', cot is used. If you don't want to use cot, you don't have to write.
```

**GSM8K**
```
python GSM8K/gsm_inference.py \
    --model_1 llama \
    --model_2 wizardlm \
    --model_3 orca \
    --API_KEY your_OpenAI_API_KEY \
    --cot    # If you write '--cot', cot is used. If you don't want to use cot, you don't have to write.
```

**MMLU**
```
python MMLU/mmlu_inference.py \
    --model_1 llama \
    --model_2 wizardlm \
    --model_3 orca \
    --API_KEY your_OpenAI_API_KEY \
    --cot    # If you write '--cot', cot is used. If you don't want to use cot, you don't have to write.
```

**Custom Inference**
```
python inference/inference.py \
    --model_1 model_you_want \
    --model_2 model_you_want \
    --model_3 model_you_want \
    --API_KEY your_OpenAI_API_KEY \
    --cot    # If you write '--cot', cot is used. If you don't want to use cot, you don't have to write.
```

You can check the result of the multi-agent debate in the folder of each task or [LLM Agora Space](https://huggingface.co/spaces/Cartinoe5930/LLMAgora).

### Evaluation

The evaluation can be performed as follows using debate response data generated through inference.
You should remember that using the same model used in inference and whether or not to use CoT must be set in the same way.

**Math**
```
python Math/math_evaulation.py \
    --model_1 llama \
    --model_2 wizardlm \
    --model_3 orca \
    --cot    # If you used 'CoT' while inference, you need to write.
```

**GSM8K**
```
python GSM8K/gsm_evaluation.py \
    --model_1 llama \
    --model_2 wizardlm \
    --model_3 orca \
    --cot    # If you used 'CoT' while inference, you need to write.
```

**MMLU**
```
python MMLU/mmlu_evaluation.py \
    --model_1 llama \
    --model_2 wizardlm \
    --model_3 orca \
    --cot    # If you used 'CoT' while inference, you need to write.
```


## Citation

```
@article{du2023improving,
  title={Improving Factuality and Reasoning in Language Models through Multiagent Debate},
  author={Du, Yilun and Li, Shuang and Torralba, Antonio and Tenenbaum, Joshua B and Mordatch, Igor},
  journal={arXiv preprint arXiv:2305.14325},
  year={2023}
}
```

```
@misc{touvron2023llama,
    title={Llama 2: Open Foundation and Fine-Tuned Chat Models}, 
    author={Hugo Touvron and Louis Martin and Kevin Stone and Peter Albert and Amjad Almahairi and Yasmine Babaei and Nikolay Bashlykov},
    year={2023},
    eprint={2307.09288},
    archivePrefix={arXiv},
}
```

```
@misc{vicuna2023,
    title = {Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\%* ChatGPT Quality},
    url = {https://lmsys.org/blog/2023-03-30-vicuna/},
    author = {Chiang, Wei-Lin and Li, Zhuohan and Lin, Zi and Sheng, Ying and Wu, Zhanghao and Zhang, Hao and Zheng, Lianmin and Zhuang, Siyuan and Zhuang, Yonghao and Gonzalez, Joseph E. and Stoica, Ion and Xing, Eric P.},
    month = {March},
    year = {2023}
}
```

```
@misc{xu2023wizardlm,
      title={WizardLM: Empowering Large Language Models to Follow Complex Instructions}, 
      author={Can Xu and Qingfeng Sun and Kai Zheng and Xiubo Geng and Pu Zhao and Jiazhan Feng and Chongyang Tao and Daxin Jiang},
      year={2023},
      eprint={2304.12244},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```
@article{falcon40b,
  title={{Falcon-40B}: an open large language model with state-of-the-art performance},
  author={Almazrouei, Ebtesam and Alobeidli, Hamza and Alshamsi, Abdulaziz and Cappelli, Alessandro and Cojocaru, Ruxandra and Debbah, Merouane and Goffinet, Etienne and Heslow, Daniel and Launay, Julien and Malartic, Quentin and Noune, Badreddine and Pannier, Baptiste and Penedo, Guilherme},
  year={2023}
}
```

```
@misc{mukherjee2023orca,
      title={Orca: Progressive Learning from Complex Explanation Traces of GPT-4}, 
      author={Subhabrata Mukherjee and Arindam Mitra and Ganesh Jawahar and Sahaj Agarwal and Hamid Palangi and Ahmed Awadallah},
      year={2023},
      eprint={2306.02707},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
