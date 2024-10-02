"""
    This will be the implementation of the web app interface for the
    simple chat bot
"""

from flask import Flask, request
from flask_cors import CORS
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ChatBot.chat_bot_response import chat_bot_response

model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

"""
    CORS allows web applications running at one domain to access 
    resources from a server at a different domain, which is normally
    blocked by browsers due to security concerns.
"""
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Hello, World!"

"""
    The expected strucutre of the data will be:

    {
    'prompt': 'message'
    }

    This is will be inside the data form the request as text. You
    then need to load it as a json, and extract the prompt.
"""
@app.route('/chatbot', methods = ['POST'])
def handle_prompt():
    # Read the data form the HTTP request body
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']

    response = chat_bot_response(model, tokenizer, input_text, conversation_history)

    conversation_history.append(input_text)
    conversation_history.append(response)

    return response

if __name__ == '__main__':
    app.run()