import gradio as gr
from llama_cpp import Llama

# Load models
llm = Llama.from_pretrained(
    repo_id="Robzy/Llama-3.2-1B-Instruct-Finetuned-q4_k_m",
    filename="unsloth.Q4_K_M.gguf",
)

llm2 = Llama.from_pretrained(
    repo_id="Robzy/Llama-3.2-1B-Instruct-Finetuned-CodeData-q4_k_m",
    filename="unsloth.Q4_K_M.gguf",
)

# Define prediction functions
def predict(message, history, model):
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


def predict2(message, history, model):
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

def predict3(message, history, model):
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



# Define ChatInterfaces
io1 = gr.ChatInterface(predict, title="4-bit")
io2 = gr.ChatInterface(predict2, title="8-bit")  # Placeholder
io3 = gr.ChatInterface(predict3, title="16-bit")
io4 = gr.ChatInterface(predict2, title="32-bit")  # Placeholder

# Dropdown and visibility mapping
chat_interfaces = {"4-bit": io1, "8-bit": io2, "16-bit": io3, "32-bit": io4}

# Define UI
with gr.Blocks() as demo:
    gr.Markdown("# Quantized Llama Comparison for Code Generation")
  
    with gr.Tab("4-bit"):
        io1.render()
    with gr.Tab("8-bit"):
        io2.render()
    with gr.Tab("16-bit"):
        io3.render()
    with gr.Tab("32-bit"):
        io4.render()


    
demo.launch()
