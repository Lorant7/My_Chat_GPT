from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Get both the pretrained model and the pretrained tokenizer for this model anme
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

# Every time you interact with the chat bot, you will pass it all of the conversation history
# where each element will be separated by "\n"
history_string = "\n".join(conversation_history)

# Example of input
input_text = "Hello, how are you doing?"

# Normally, tokenizer have a methdo called "encode_plus" that is used to tokenize or vectorize
# the input to the model.
#
# Here, I am passing it the history and the current input and telling it to return it as 
# a PyTorch tensor using "return_tensor = "pt""
inputs  = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
print(inputs)