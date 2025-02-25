---
title: quantized-LLM comparison 
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
short_descriptions: Fine-tuned Llama-3.2-1B-Instruct with different quantizations
---


### [HuggingFace Space with Quantized LLMs](https://huggingface.co/spaces/Robzy/llm)

**Baseline model**: Llama-3.2-1B-Instruct with 4-bit quantization

**Training infrastracture**:
* Google Colab with NVIDIA Tesla T4 GPU
* Finetuning with parameter-effecient finetuning (PEFT) by low-rank adaption (LORA) using Unsloth and HuggingFace's supervised finetuning libraries. 
* Weight & Biases for model training monitoring and model checkpointing. Checkpointing every 10 steps.

**Finetuning details**
* Data centric approach:
* We used datacentric approach to finetune the model on code generation data to improve it's performance on genrating code based on instructions
* We used unsloth/Llama-3.2-1B-Instruct as the base model for all experiments
* We finetuned the model on iamtarun/python_code_instructions_18k_alpaca which contains problem descriptions and code in python language.
* We also finetuned the model on a larger dataset for code generation: iamtarun/code_instructions_120k_alpaca
* We saved the model in different quantized forms: "q4_k_m", "q8_0", "q5_k_m"

**Hyperparameters**
* num_train_epochs = 1
* save_steps=10 
* max_steps = 0
* learning_rate = 2e-4,
* logging_steps = 1,
* weight_decay = 0.01,
* lr_scheduler_type = "linear",

**Datasets**:
* [Code instructions Alpaca 18k](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)
* [Code instructions Alpaca 120k](https://huggingface.co/datasets/iamtarun/code_instructions_120k_alpaca)
