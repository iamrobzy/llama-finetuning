from llama_cpp import Llama
import gradio as gr

llm = Llama.from_pretrained(
	repo_id="Robzy/Llama-3.2-1B-Instruct-Finetuned-q4_k_m",
	filename="unsloth.Q4_K_M.gguf",
)

llm2 = Llama.from_pretrained(
    repo_id="Robzy/Llama-3.2-1B-Instruct-Finetuned-16bit",
    filename="unsloth.F16.gguf",
)

def predict(message, history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for user_message, bot_message in history:
        if user_message:
            messages.append({"role": "user", "content": user_message})
        if bot_message:
            messages.append({"role": "assistant", "content": bot_message})
    messages.append({"role": "user", "content": message})
    
    response = ""
    for chunk in llm.create_chat_completion(
        stream=True,
        messages=messages,
    ):
        part = chunk["choices"][0]["delta"].get("content", None)
        if part:
            response += part
        yield response


def predict2(message, history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for user_message, bot_message in history:
        if user_message:
            messages.append({"role": "user", "content": user_message})
        if bot_message:
            messages.append({"role": "assistant", "content": bot_message})
    messages.append({"role": "user", "content": message})
    
    response = ""
    for chunk in llm2.create_chat_completion(
        stream=True,
        messages=messages,
    ):
        part = chunk["choices"][0]["delta"].get("content", None)
        if part:
            response += part
        yield response


chat1 = gr.ChatInterface(predict, title="4-bit")
chat2 = gr.ChatInterface(predict2, title="16-bit")
chat3 = gr.ChatInterface(predict2, title="16-bit")

def update_chat(value):
    if value == "4-bit":
        chat1.render(visible=True)
        chat2.render(visible=False)
        chat3.render(visible=False)
    elif value == "16-bit":
        chat1.render(visible=False)
        chat2.render(visible=True)
        chat3.render(visible=False)
    else:
        chat1.render(visible=False)
        chat2.render(visible=False)
        chat3.render(visible=True)

with gr.Blocks() as demo:

    gr.Markdown("# Quantized Llama Comparison for Code Generation")
    dropdown = gr.Dropdown(["4-bit", "16-bit", "32-bit"], label="Choose model version", value="4-bit")
    dropdown.change(fn=update_chat, inputs=dropdown, outputs=[chat1, chat2, chat3])

demo.launch()