import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_PATH = os.getenv("PYTHONPATH", "/home/sachdeva/projects/exp_calibration/")

if __name__ == '__main__':
    model_path = BASE_PATH+"t5-base-nq-short-qa-10ep/checkpoint-107990"
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    sep_token = ">>"
    question = "What is the name of the stadium where the Super Bowl was held?"
    context = "The Super bowl was held at the Levi's Stadium in the New York City."
    prepared_input = f"{question} {sep_token} {context}"
    features = tokenizer(prepared_input, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    # print(features)
    outputs = model.generate(**features, max_length=128, num_beams=2)
    dec_preds = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("The prediction is: ", dec_preds)
