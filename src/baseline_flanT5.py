
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def answer_question(question):
    input_text = f"Question: {question}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=80)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    q = input("Enter your question: ")
    print("Answer:", answer_question(q))
