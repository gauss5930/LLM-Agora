# LLM-Agora
LLM Agora, debating between open-source LLMs to refine the answers

The **LLM Agora** 🗣️🏦 aims to improve the quality of open-source LMs' responses through debate & revision introduced in [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325).

Do you know that? 🤔 **LLMs can also improve their responses by debating with other LLMs**! 😮 We applied this concept to several open-source LMs to verify that the open-source model, not the proprietary one, can sufficiently improve the response through discussion. 🤗
For more details, please refer to the [GitHub Repository](https://github.com/gauss5930/LLM-Agora).

You can use LLM Agora with your own questions if the response of open-source LM is not satisfactory and you want to improve the quality!
The Math, GSM8K, and MMLU Tabs show the results of the experiment(Llama2, WizardLM2, Orca2), and for inference, please use the 'Inference' tab.

Here's how to use LLM Agora!

1. Choose just 3 models! Neither more nor less!
2. Check the CoT box if you want to utilize the Chain-of-Thought while inferencing.
3. Please fill in your OpenAI API KEY, it will be used to use ChatGPT to summarize the responses.
4. Type your question to Question box and click the 'Submit' button! If you do so, LLM Agora will show you improved answers! 🤗

For more information, please check '※ Specific information about LLM Agora' at the bottom of the page.

## Inference

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
python GSM8K/gsm_inference.py \
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

## Evaluation

### paper version

You should use the same model used in inference.

**GSM8K**
```
python GSM8K/gsm_evaluation.py \
    --model_1 llama \
    --model_2 alpaca \
    --model_3 vicuna \
    --cot True_or_False \
```

**MMLU**
```
python MMLU/mmlu_evaluation.py \
    --model_1 llama \
    --model_2 alpaca \
    --model_3 vicuna \
    --cot True_or_False \
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
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
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
@misc{koala_blogpost_2023,
  author = {Xinyang Geng and Arnav Gudibande and Hao Liu and Eric Wallace and Pieter Abbeel and Sergey Levine and Dawn Song},
  title = {Koala: A Dialogue Model for Academic Research},
  howpublished = {Blog post},
  month = {April},
  year = {2023},
  url = {https://bair.berkeley.edu/blog/2023/04/03/koala/},
  urldate = {2023-04-03}
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
@article{xu2023baize,
  title={Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data},
  author={Xu, Canwen and Guo, Daya and Duan, Nan and McAuley, Julian},
  journal={arXiv preprint arXiv:2304.01196},
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

```
@software{OpenOrca_Preview1,
  title = {OpenOrca_Preview1: A LLaMA-13B Model Fine-tuned on Small Portion of OpenOrcaV1 Dataset},
  author = {Wing Lian and Bleys Goodson and Eugene Pentland and Austin Cook and Chanvichet Vong and "Teknium"},
  year = {2023},
  publisher = {HuggingFace},
  journal = {HuggingFace repository},
  howpublished = {\url{https://https://huggingface.co/Open-Orca/OpenOrca-Preview1-13B},
}
```

```
@article{textbooks2,
  title={Textbooks Are All You Need II: \textbf{phi-1.5} technical report},
  author={Li, Yuanzhi and Bubeck, S{\'e}bastien and Eldan, Ronen and Del Giorno, Allie and Gunasekar, Suriya and Lee, Yin Tat},
  journal={arXiv preprint arXiv:2309.05463},
  year={2023}
}
```