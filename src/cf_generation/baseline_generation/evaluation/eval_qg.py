import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_PATH = os.getenv("PYTHONPATH", "/home/sachdeva/projects/exp_calibration/")

if __name__ == "__main__":
    model_path = BASE_PATH + "t5-base-nq-short-qg-10ep/checkpoint-109150"
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    hl_token = "<hl>"
    context = f"The Super bowl was held at the Levi's Stadium in the {hl_token} New York {hl_token}."
    prepared_input = f"generate question: {context}"
    features = tokenizer(
        prepared_input,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    # print(features)
    outputs = model.generate(**features, max_length=128, num_beams=2)
    dec_preds = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("The prediction is: ", dec_preds)
