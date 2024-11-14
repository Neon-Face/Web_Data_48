from llama_cpp import Llama
model_path = "models/llama-2-7b.Q4_K_M.gguf"

# If you want to use larger models...
#model_path = "models/llama-2-13b.Q4_K_M.gguf"
# All models are available at https://huggingface.co/TheBloke. Make sure you download the ones in the GGUF format
#The goal of the assignment is to implement a program that post-processes the output of large language models (like ChatGPT, etc.) to improve the quality of its answers.
question = "What is the capital of The Netherlands? "
llm = Llama(model_path=model_path, verbose=False)
print("Asking the question \"%s\" to %s (wait, it can take some time...)" % (question, model_path))
output = llm(
      question, # Prompt
      max_tokens=16, # Generate up to 32 tokens
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
)
print("Here is the output")
print(output['choices'])



import re


def clean_text(text):
    """
    Cleans the input text by removing noise, redundant text, and special characters.
    Args:
        text (str): The input text to clean.
    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# Clean the input question
cleaned_question = clean_text(question)
print("Cleaned Question:", cleaned_question)



# Query the Llama model
print(f"Asking the question: '{cleaned_question}' (wait, it can take some time...)")
output = llm(
    cleaned_question,
    max_tokens=16,
    stop=["Q:", "\n"],
    echo=True
)

# Process and print model output
raw_output = output['choices'][0]['text']
print("Raw output:", raw_output)