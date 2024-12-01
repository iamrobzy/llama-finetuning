from llama_cpp import Llama
import gradio as gr

llm = Llama.from_pretrained(
	repo_id="Robzy/Llama-3.2-1B-Instruct-Finetuned-q4_k_m",
	filename="unsloth.Q4_K_M.gguf",
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

demo = gr.ChatInterface(predict)

if __name__ == "__main__":
    demo.launch()
