---
title: Llm
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
---


### [HuggingFace Space with Quantized LLMs](https://huggingface.co/spaces/Robzy/llm)

**Baseline model**: Llama-3.2-1B-Instruct with 4-bit quantization

**Training infrastracture**:
* Google Colab with NVIDIA Tesla T4 GPU
* Finetuning with parameter-effecient finetuning (PEFT) by low-rank adaption (LORA) using Unsloth and HuggingFace's supervised finetuning libraries. 
* Weight & Biases for model training monitoring and model checkpointing. Checkpointing every 10 steps.

**Finetuning details**
* Data centric approach: 

**Datasets**:
* [Code instructions Alpaca 120k](https://huggingface.co/datasets/iamtarun/code_instructions_120k_alpaca)
