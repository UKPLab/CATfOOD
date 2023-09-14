import os

from transformers import AutoModelForCausalLM, T5Tokenizer, AutoTokenizer
import torch
import jsonlines
from tqdm import tqdm
from datasets import load_dataset

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"
# BASE_PATH = "/home/sachdeva/projects/ukp/exp_calibration/"


def save_to_disk(data, file_name):
    with jsonlines.open(file_name, "a") as writer:
        for example in tqdm(data, total=len(data), desc="Saving samples ... "):
            writer.write(example)


if __name__ == "__main__":

    # load squad data
    dataset = load_dataset("squad", "plain_text")
    train_data = dataset["train"]
    squad_data = [
        sample
        for sample in tqdm(
            train_data, total=len(train_data), desc="Loading SQuAD data ... "
        )
    ]

    c = 0
    examples = []

    prompt = (
        "Generate a fluent and answerable question from the given context. Ensure that the answer "
        "is a span in the context and is less than 10 words."
    )

    model_name = "EleutherAI/gpt-j-6B"
    model_identifier = model_name.split("/")[-1]
    save_path = BASE_PATH + f"src/data/squad/{model_identifier}_qg"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with jsonlines.open(
        BASE_PATH
        + "src/data/squad/squad_counterfactuals_noise_min_filtered_final_2.jsonl"
    ) as reader:
        for example in tqdm(reader):
            c += 1
            id = example["id"].split("_")[0]
            context = example["context"]
            orig_example = [sample for sample in squad_data if sample["id"] == id][0]
            # print(orig_example)
            orig_context = orig_example["context"]
            orig_question = orig_example["question"]
            orig_answer = orig_example["answers"]

            input = (
                f"{prompt} \nContext: {orig_context} \nQuestion: {orig_question} "
                f"\n{prompt} \nContext: {context} "
                f"\nQuestion: "
            )

            # print(input)
            inputs = tokenizer(input, return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(inputs, max_length=200)
            result = {
                "id": example["id"],
                "question": tokenizer.decode(outputs[0], skip_special_tokens=True),
                "context": context,
            }
            # print(result)
            examples.append(result)
            if c % 5000 == 0:
                save_to_disk(
                    examples,
                    f"{save_path}/counterfactual_questions_{model_identifier}_{c}.jsonl",
                )
                examples = []

        # save the remaining examples
        if examples:
            save_to_disk(
                examples,
                f"{save_path}/counterfactual_questions_{model_identifier}_{c}.jsonl",
            )
