# ai-chat-box
import openai
import tkinter as tk
from tkinter import scrolledtext

# Set up OpenAI API key
openai.api_key = "your_openai_api_key_here"

class AIChatBox:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Chat Box")
        
        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', width=60, height=20)
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        
        # User input field
        self.user_input = tk.Entry(root, width=50)
        self.user_input.grid(row=1, column=0, padx=10, pady=10)
        self.user_input.bind("<Return>", self.send_message)
        
        # Send button
        self.send_button = tk.Button(root, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)
    
    def send_message(self, event=None):
        user_message = self.user_input.get().strip()
        if user_message:
            self.display_message("You", user_message)
            self.user_input.delete(0, tk.END)
            self.get_ai_response(user_message)
    
    def display_message(self, sender, message):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"{sender}: {message}\n")
        self.chat_display.yview(tk.END)
        self.chat_display.config(state='disabled')
    
    def get_ai_response(self, user_message):
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"The following is a conversation with an AI assistant.\n\nUser: {user_message}\nAI:",
                max_tokens=150,
                temperature=0.7
            )
            ai_message = response.choices[0].text.strip()
            self.display_message("AI", ai_message)
        except Exception as e:
            self.display_message("AI", "Sorry, I couldn't process your request.")

if __name__ == "__main__":
    root = tk.Tk()
    chat_box = AIChatBox(root)
    root.mainloop()
    # chatbot.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import pickle

# Load model and tokenizer
MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Memory file
MEMORY_FILE = "chat_memory.pkl"
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "rb") as f:
        chat_history_ids = pickle.load(f)
else:
    chat_history_ids = None

def chatbot_response(user_input, chat_memory=None):
    global chat_history_ids

    # Encode input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)

    # Append history
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )

    # Decode
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Save chat memory
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(chat_history_ids, f)

    return response

# Reset memory
def reset_chat():
    global chat_history_ids
    chat_history_ids = None
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
    return "Chat reset."

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Simple AI Chatbot using DialoGPT")
    
    with gr.Row():
        with gr.Column(scale=8):
            chatbot_output = gr.Textbox(lines=5, label="Chatbot Response")
        with gr.Column(scale=4):
            reset_button = gr.Button("🔄 Reset Chat")

    with gr.Row():
        user_input = gr.Textbox(placeholder="Type your message here...", label="You")
        submit_button = gr.Button("Send")

    # Functional link
    submit_button.click(fn=chatbot_response, inputs=user_input, outputs=chatbot_output)
    reset_button.click(fn=reset_chat, outputs=chatbot_output)

# Launch interface
demo.launch()

