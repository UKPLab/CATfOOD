import os

from transformers import AutoModelForCausalLM, AutoTokenizer


import torch
import jsonlines
from tqdm import tqdm
from datasets import load_dataset

from src.few_shot.utils import save_to_disk

BASE_PATH="/storage/xyz/work/anon/research_projects/exp_calibration/"
# BASE_PATH = "/home/anon/projects/xyz/exp_calibration/"



if __name__ == '__main__':

    # load squad data
    dataset = load_dataset("squad", "plain_text")
    train_data = dataset["train"]
    squad_data = [sample for sample in tqdm(train_data, total=len(train_data), desc="Loading SQuAD data ... ")]

    c = 0
    examples = []

    # prompt = "Generate a fluent and answerable question from the given context. Ensure that the answer " \
    #          "is a span in the context and is less than 10 words."

    # prompt = "Given the context below, generate a fluent question that can be answered from it."  # simple
    # prompt = "Using the context provided, produce a well-crafted question that can be answered from it."  # vague
    # prompt = "From the given context, create a clear and concise question that can be answered from it."  # vague
    # prompt = "Given the context below, create a specific question that can be answered using only information " \
    #          "directly stated in the passage."
    # prompt = "Using the given context, create a question that can be answered by selecting a relevant span from the text."
    # prompt = "From the provided context, create a question that can be accurately answered by extracting " \
    #          "information directly from the context."
    # prompt = "Construct a question that can be answered by selecting a key piece of information from the given context."
    prompt = "Generate a fluent and answerable question from the given context. Ensure that the answer " \
             "is a span in the context and is less than 10 words."

    model_name = "facebook/opt-13b"
    model_identifier = model_name.split("/")[-1]
    save_path = BASE_PATH + f"src/data/squad/{model_identifier}_qg"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    with jsonlines.open(BASE_PATH + "src/data/squad/squad_counterfactuals_noise_min_filtered_final_2.jsonl") as reader:
        for example in tqdm(reader):
            c += 1
            id = example["id"].split("_")[0]
            context = example["context"]
            orig_example = [sample for sample in squad_data if sample["id"] == id][0]
            # print(orig_example)
            orig_context = orig_example["context"]
            orig_question = orig_example["question"]
            orig_answer = orig_example["answers"]

            # input = f"{prompt} \nContext: {context} \nQuestion: "
            input = f"{prompt} \n\nContext: {context} \n\nQuestion: "
            #
            # input = f"{prompt} \nContext: {orig_context} \nQuestion: {orig_question} \nAnswer span: {orig_answer['text'][0]} " \
            #         f"\n{prompt} \nContext: {context} " \
            #         f"\nQuestion: \nAnswer span: "

            # print(input)
            inputs = tokenizer(input, return_tensors="pt").to("cuda")
            generated_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=50
            )
            outputs = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(outputs)

            if c == 2:
                break
            # break
            # result = {
            #     "id": example["id"],
            #     "question": tokenizer.decode(outputs[0], skip_special_tokens=True),
            #     "context": context
            # }
            # print(result)
        #     examples.append(result)
        #     if c % 5000 == 0:
        #         save_to_disk(
        #             examples,
        #             f"{save_path}/counterfactual_questions_{model_identifier}_{c}.jsonl"
        #         )
        #         examples = []
        #
        # # save the remaining examples
        # if examples:
        #     save_to_disk(
        #         examples,
        #         f"{save_path}/counterfactual_questions_{model_identifier}_{c}.jsonl"
        #     )
