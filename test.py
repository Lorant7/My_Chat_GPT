from ChatBot.chat_bot_response import chat_bot_response
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = "Hia"

conversation_history = []

response = chat_bot_response(model, tokenizer, input_text, conversation_history)
print(response)
