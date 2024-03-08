<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="images/logo.svg" width="60%" alt="DeepSeek LLM" />
</div>
<hr>
<div align="center">

  <a href="https://www.deepseek.com/" target="_blank">
    <img alt="Homepage" src="images/badge.svg" />
  </a>
  <a href="https://chat.deepseek.com/" target="_blank">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20LLM-536af5?color=536af5&logoColor=white" />
  </a>
  <a href="https://huggingface.co/deepseek-ai" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" />
  </a>

</div>


<div align="center">

  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" />
  </a>
  <a href="images/qr.jpeg" target="_blank">
    <img alt="Wechat" src="https://img.shields.io/badge/WeChat-DeepSeek%20AI-brightgreen?logo=wechat&logoColor=white" />
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" />
  </a>

</div>

<div align="center">

  <a href="LICENSE-CODE">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5de53">
  </a>
  <a href="LICENSE-MODEL">
    <img alt="Model License" src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53">
  </a>
</div>


<p align="center">
  <a href="#2-model-downloads">Model Download</a> |
  <a href="#5-quick-start">Quick Start</a> |
  <a href="#3-evaluation-results">Evaluation Results</a> |
  <a href="#8-license">License</a> |
  <a href="#9-citation">Citation</a>
</p>


<p align="center">
  <a href="https://arxiv.org/abs/2401.02954"><b>Paper Link</b>👁️</a>
</p>


## 1. Introduction

Introducing DeepSeek VL,

 - **Superior General Capabilities:** DeepSeek .

 - **Proficient in Coding and Math:** DeepSeek .

 - **Mastery in Chinese Language:** Based on our evaluation,.

## 2. Model Downloads

We release the DeepSeek-VL family, including 1.3B-base, 1.3B-chat, 7b-base and 7b-chat models, to the public.
To support a broader and more diverse range of research within both academic and commercial communities.
Please note that the use of this model is subject to the terms outlined in [License section](#8-license). Commercial usage is
permitted under these terms.

### Huggingface

| Model                 | Sequence Length | Download                                                                    |
|-----------------------|-----------------|-----------------------------------------------------------------------------|
| DeepSeek-VL-1.3B-base | 4096            | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-base) |
| DeepSeek-VL-1.3B-chat | 4096            | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat) |
| DeepSeek-VL-7B-base   | 4096            | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/deepseek-vl-7b-base)   |
| DeepSeek-VL-7B-chat   | 4096            | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)   |

# 3. Evaluation Results

### Base Model

We evaluate our models and some baseline models on a series of representative benchmarks, both in English and Chinese. More results can be found in the [`evaluation`](evaluation) folder. In this part, the evaluation results we report are based on the internal, non-open-source hai-llm evaluation framework. Please note that there may be slight discrepancies when using the converted HuggingFace models.

|      model      | Hella<br>Swag | Trivia<br>QA |  MMLU  | GSM8K  | Human<br>Eval |  BBH   | CEval  | CMMLU  | Chinese<br>QA |
|:---------------:|:-------------:|:------------:|:------:|:------:|:-------------:|:------:|:------:|:------:|:-------------:|
|                 |    0-shot     |    5-shot    | 5-shot | 8-shot |    0-shot     | 3-shot | 5-shot | 5-shot |    5-shot     |
| LLaMA-2<br>-7B  |     75.6      |     63.8     |  45.8  |  15.5  |     14.6      |  38.5  |  33.9  |  32.6  |     21.5      |
| LLaMA-2<br>-70B |     84.0      |     79.5     |  69.0  |  58.4  |     28.7      |  62.9  |  51.4  |  53.1  |     50.2      |
| DeepSeek LLM<br>7B Base|     75.4      |     59.7     |  48.2  |  17.4  |     26.2      |  39.5  |  45.0  |  47.2  |     78.0      |
| DeepSeek LLM<br>67B Base|     84.0      |     78.9     |  71.3  |  63.4  |     42.7      |  68.7  |  66.1  |  70.8  |     87.6      |

**Note:** ChineseQA is an in-house benchmark, inspired by TriviaQA.

### Chat Model

#### Never Seen Before Exam


## 4. Pre-Training Details

### Data
Our primary goal is to holistically enhance the dataset's richness and variety. To achieve this, we've implemented multiple methods and established a distributed, frequent-checkpointing batch processing system, named "cc_cleaner", to bolster our data pipeline.

Our minimal viable solution departs from RefinedWeb + CCNet. We greatly appreciate their selfless dedication to the research of AGI.

We have also significantly incorporated deterministic randomization into our data pipeline. This approach enables us to continuously enhance our data throughout the lengthy and unpredictable training process.

- **Data Composition**: Our training data comprises a diverse mix of Internet text, math, code, books, and self-collected data respecting robots.txt. In addition to the diverse content, we place a high priority on personal privacy and copyright protection. All content containing personal information or subject to copyright restrictions has been removed from our dataset.

- **Dataset Pruning**: Our system employs heuristic rules and models to refine our training data. Our filtering process removes low-quality web data while preserving precious low-resource knowledge. It aims to improve overall corpus quality and remove harmful or toxic content.

- **Deduplication**: Our advanced deduplication system, using MinhashLSH, strictly removes duplicates both at document and string levels. This rigorous deduplication process ensures exceptional data uniqueness and integrity, especially crucial in large-scale datasets.

### Pre-Training
DeepSeek LM models use the same architecture as LLaMA, an auto-regressive transformer decoder model. The 7B model uses Multi-Head attention (MHA) while the 67B model uses Grouped-Query Attention (GQA).

We pre-trained DeepSeek language models on a vast dataset of 2 trillion tokens, with a sequence length of 4096 and AdamW optimizer. The 7B model's training involved a batch size of 2304 and a learning rate of 4.2e-4 and the 67B model was trained with a batch size of 4608 and a learning rate of 3.2e-4. We employ a multi-step learning rate schedule in our training process. The learning rate begins with 2000 warmup steps, and then it is stepped to 31.6% of the maximum at 1.6 trillion tokens and 10% of the maximum at 1.8 trillion tokens.

We release the training loss curve and several benchmark metrics curves, as detailed below.

<div align="center">
  <img src="images/pretrain_loss.png" alt="result" width="70%">
</div>

<div align="center">
  <img src="images/pretrain_metric.png" alt="result" width="80%">
</div>

## 5. Quick Start

### Installation

On the basis of `Python >= 3.8` environment, install the necessary dependencies by running the following command:

```shell
pip install -r requirements.txt -e .
```

### Inference with Huggingface's Transformers

You can directly employ [Huggingface's Transformers](https://github.com/huggingface/transformers) for model inference.

**Simple Inference Example**

```python
import torch
from transformers import AutoModelForCausalLM

from deepseek_vlm.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vlm.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>Describe each stage of this image.",
        "images": ["./images/training_pipelines.png"]
    },
    {
        "role": "Assistant",
        "content": ""
    }
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True
).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)
```

**CLI Chat**
```bash
python cli_chat.py --model_path deepseek-ai/deepseek-vl-7b-chat
```

Avoiding the use of the provided function `apply_chat_template`, you can also interact with our model following the sample template. Note that `messages` should be replaced by your input.

```
User: {messages[0]['content']}

Assistant: {messages[1]['content']}<｜end▁of▁sentence｜>User: {messages[2]['content']}

Assistant:
```

**Note:** By default (`add_special_tokens=True`), our tokenizer automatically adds a `bos_token` (`<｜begin▁of▁sentence｜>`) before the input text. Additionally, since the system prompt is not compatible with this version of our models, we DO NOT RECOMMEND including the system prompt in your input.

### Inference with vLLM

You can also employ [vLLM](https://github.com/vllm-project/vllm) for high-throughput inference.

**Text Completion**

```python
from vllm import LLM, SamplingParams

tp_size = 4 # Tensor Parallelism
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
model_name = "deepseek-ai/deepseek-llm-67b-base"
llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=tp_size)

prompts = [
    "If everyone in a country loves one another,",
    "The research should also focus on the technologies",
    "To determine if the label is correct, we need to"
]
outputs = llm.generate(prompts, sampling_params)

generated_text = [output.outputs[0].text for output in outputs]
print(generated_text)
```

**Chat Completion**

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tp_size = 4 # Tensor Parallelism
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
model_name = "deepseek-ai/deepseek-llm-67b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=tp_size)

messages_list = [
    [{"role": "user", "content": "Who are you?"}],
    [{"role": "user", "content": "What can you do?"}],
    [{"role": "user", "content": "Explain Transformer briefly."}],
]
# Avoid adding bos_token repeatedly
prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]

sampling_params.stop = [tokenizer.eos_token]
outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

generated_text = [output.outputs[0].text for output in outputs]
print(generated_text)
```

## 6. FAQ

### Could You Provide the tokenizer.model File for Model Quantization?

DeepSeek LLM utilizes the [HuggingFace Tokenizer](https://huggingface.co/docs/tokenizers/index) to implement the Byte-level BPE algorithm, with specially designed pre-tokenizers to ensure optimal performance. Currently, there is no direct way to convert the tokenizer into a SentencePiece tokenizer. We are contributing to the open-source quantization methods facilitate the usage of HuggingFace Tokenizer.

#### GGUF(llama.cpp)

We have submitted a [PR](https://github.com/ggerganov/llama.cpp/pull/4070) to the popular quantization repository [llama.cpp](https://github.com/ggerganov/llama.cpp) to fully support all HuggingFace pre-tokenizers, including ours.

While waiting for the PR to be merged, you can generate your GGUF model using the following steps:

```bash
git clone https://github.com/DOGEwbx/llama.cpp.git
cd llama.cpp
git checkout regex_gpt2_preprocess
# set up the environment according to README
make
python3 -m pip install -r requirements.txt
# generate GGUF model
python convert-hf-to-gguf.py <MODEL_PATH> --outfile <GGUF_PATH> --model-name deepseekllm
# use q4_0 quantization as an example
./quantize <GGUF_PATH> <OUTPUT_PATH> q4_0
./main -m <OUTPUT_PATH> -n 128 -p <PROMPT>
```
#### GPTQ(exllamav2)

`UPDATE:`[exllamav2](https://github.com/turboderp/exllamav2) has been able to support HuggingFace Tokenizer. Please pull the latest version and try out.

### GPU Memory Usage

We profile the peak memory usage of inference for 7B and 67B models at different batch size and sequence length settings.

For DeepSeek LLM 7B, we utilize **1 NVIDIA A100-PCIE-40GB GPU** for inference.

<table><thead><tr><th rowspan="2">Batch Size</th><th colspan="5">Sequence Length</th></tr><tr><th>256</th><th>512</th><th>1024</th><th>2048</th><th>4096</th></tr></thead><tbody><tr><td>1</td><td>13.29 GB</td><td>13.63 GB</td><td>14.47 GB</td><td>16.37 GB</td><td>21.25 GB</td></tr><tr><td>2</td><td>13.63 GB</td><td>14.39 GB</td><td>15.98 GB</td><td>19.82 GB</td><td>29.59 GB</td></tr><tr><td>4</td><td>14.47 GB</td><td>15.82 GB</td><td>19.04 GB</td><td>26.65 GB</td><td>OOM</td></tr><tr><td>8</td><td>15.99 GB</td><td>18.71 GB</td><td>25.14 GB</td><td>35.19 GB</td><td>OOM</td></tr><tr><td>16</td><td>19.06 GB</td><td>24.52 GB</td><td>37.28 GB</td><td>OOM</td><td>OOM</td></tr></tbody></table>

For DeepSeek LLM 67B, we utilize **8 NVIDIA A100-PCIE-40GB GPUs** for inference.

<table><thead><tr><th rowspan="2">Batch Size</th><th colspan="5">Sequence Length</th></tr><tr><th>256</th><th>512</th><th>1024</th><th>2048</th><th>4096</th></tr></thead><tbody><tr><td>1</td><td>16.92 GB</td><td>17.11 GB</td><td>17.66 GB</td><td>20.01 GB</td><td>33.23 GB</td></tr><tr><td>2</td><td>17.04 GB</td><td>17.28 GB</td><td>18.55 GB</td><td>25.27 GB</td><td>OOM</td></tr><tr><td>4</td><td>17.20 GB</td><td>17.80 GB</td><td>21.28 GB</td><td>33.71 GB</td><td>OOM</td></tr><tr><td>8</td><td>17.59 GB</td><td>19.25 GB</td><td>25.69 GB</td><td>OOM</td><td>OOM</td></tr><tr><td>16</td><td>18.17 GB</td><td>21.69 GB</td><td>34.54 GB</td><td>OOM</td><td>OOM</td></tr></tbody></table>

## 7. Limitation

While DeepSeek LLMs have demonstrated impressive capabilities, they are not without their limitations. Here are some potential drawbacks of such models:

1. Over-reliance on training data: These models are trained on vast amounts of text data, which can introduce biases present in the data. They may inadvertently generate biased or discriminatory responses, reflecting the biases prevalent in the training data.

2. Hallucination: The model sometimes generates responses or outputs that may sound plausible but are factually incorrect or unsupported. This can occur when the model relies heavily on the statistical patterns it has learned from the training data, even if those patterns do not align with real-world knowledge or facts.

3. Repetition: The model may exhibit repetition in their generated responses. This repetition can manifest in various ways, such as repeating certain phrases or sentences, generating redundant information, or producing repetitive structures in the generated text. This issue can make the output of LLMs less diverse and less engaging for users.

## 8. License

This code repository is licensed under [the MIT License](https://github.com/deepseek-ai/DeepSeek-LLM/blob/HEAD/LICENSE-CODE). The use of DeepSeek LLM Base/Chat models is subject to [the Model License](https://github.com/deepseek-ai/DeepSeek-LLM/blob/HEAD/LICENSE-MODEL). DeepSeek LLM series (including Base and Chat) supports commercial use.

## 9. Citation

```
@article{deepseek-llm,
  author = {DeepSeek-AI},
  title = {DeepSeek LLM: Scaling Open-Source Language Models with Longtermism},
  journal = {arXiv preprint arXiv:2401.02954},
  year = {2024},
  url = {https://github.com/deepseek-ai/DeepSeek-LLM}
}
```

## 10. Contact

If you have any questions, please raise an issue or contact us at [service@deepseek.com](mailto:service@deepseek.com).
