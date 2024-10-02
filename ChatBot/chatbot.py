"""
    Implementation of a simple chat bot using Facebook's blender bot mode
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def run_chat_bot(model_name):
    """
        Recursively prompts the user for text to send to the chat bot and prints
        the chat bot's reponse
    """
    model_name = "facebook/blenderbot-400M-distill"

    # Get both the pretrained model and the pretrained tokenizer for this model anme
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    conversation_history = []

    while True:
        history_string = "\n".join(conversation_history)

        input_text = input("> ")

        inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

        outputs = model.generate(**inputs)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

        print(response)

        conversation_history.append(input_text)
        conversation_history.append(response)
    