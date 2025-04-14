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
