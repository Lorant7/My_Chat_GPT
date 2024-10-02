
"""
    This script holds a function that gets the resopnse from the chat bot
"""
def chat_bot_response(model, tokenizer, input_text, conversation_history):
    """
        Takes in the model, tokenizer, text inputted and conversations history
        to return the response form the chat bot
    """
    history_string = "\n".join(conversation_history)

    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    outputs = model.generate(**inputs)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

    conversation_history.append(input_text)
    conversation_history.append(response)

    return response
