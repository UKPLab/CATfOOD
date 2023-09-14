from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
import torch
import os
import jsonlines
import argparse
from tqdm import tqdm
from datasets import load_dataset

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"
# BASE_PATH = "/home/sachdeva/projects/ukp/exp_calibration/"


def save_to_disk(data, file_name):
    with jsonlines.open(file_name, "a") as writer:
        for example in tqdm(data, total=len(data), desc="Saving samples ... "):
            writer.write(example)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Set the seed
    torch.manual_seed(args.seed)

    # load squad data
    dataset = load_dataset("squad", "plain_text")
    train_data = dataset["train"]
    squad_data = [
        sample
        for sample in tqdm(
            train_data, total=len(train_data), desc="Loading SQuAD data ... "
        )
    ]

    current_files = []

    c = 0
    # prompt = "Given the following question and context, provide an answer " \
    #          "that best fits the question. Ensure that the answer " \
    #          "is a span in the context."
    prompt = (
        "Answer the question based on the context below. If the question cannot be answered "
        "using the information provided, then answer with 'I don't know'."
    )

    # to test
    # prompt = "Your task is to answer a question based on the given context. If the information provided " \
    #          "is insufficient to answer the question, please respond with 'I don't know.' Your response " \
    #          "should be clear and concise, providing only relevant information necessary to answer the question. " \
    #          "Please note that you should make every effort to provide an accurate and complete response based " \
    #          "on the available information."

    model_name = "google/flan-t5-xxl"
    model_identifier = model_name.split("/")[-1]

    save_path = (
        BASE_PATH
        + f"src/data/squad/noise_filter_{model_identifier}_qa_eval_seed_{args.seed}"
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = T5ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # all_files = os.listdir(BASE_PATH + f"src/data/squad/few_shot_{model_identifier}_qg_temp_0.7/")
    # files = [file for file in all_files if file not in current_files]
    # print(files)
    files = [
        "counterfactual_samples_Llama-2-13b-chat-hf_flan-t5-xxl_context_filtered_complete.jsonl"
    ]

    for file in files:
        c += 1
        # if c<=9:
        #     continue
        print(f"Processing file: {file}")
        examples = []
        with jsonlines.open(BASE_PATH + f"src/data/squad/" + file) as reader:
            # with jsonlines.open(BASE_PATH + f"src/data/squad/few_shot_{model_identifier}_qg_temp_0.7/" + file) as reader:
            for example in tqdm(reader):
                id = example["id"].split("_")[0]
                context = example["context"]
                question = example["question"]
                tokens_to_remove = ["[", "'", '"', "]"]
                # Create a translation table that maps each unwanted token to None
                translator = str.maketrans({token: None for token in tokens_to_remove})
                question = question.translate(translator).strip()

                orig_example = [sample for sample in squad_data if sample["id"] == id][
                    0
                ]

                orig_context = orig_example["context"]
                orig_question = orig_example["question"]
                orig_answer = orig_example["answers"]

                # input = f"{prompt} \nQuestion: {orig_question} \nContext: {orig_context} " \
                #         f"\nAnswer: {orig_answer['text'][0]}" \
                #         f"\n{prompt} \nQuestion: {question}  \nContext: {context} " \
                #         f"\nAnswer: "

                input = (
                    f"{prompt} \nContext: {orig_context} \nQuestion: {orig_question} "
                    f"\nAnswer: {orig_answer['text'][0]}"
                    f"\n{prompt} \nContext: {context} \nQuestion: {question} "
                    f"\nAnswer: "
                )

                inputs = tokenizer(input, return_tensors="pt").input_ids.to("cuda")
                outputs = model.generate(
                    inputs,
                    max_length=200,
                    temperature=0.7,
                    num_beams=1,
                    num_return_sequences=1,
                )
                result = {
                    "id": example["id"],
                    "question": question,
                    "context": context,
                    "answer": tokenizer.decode(outputs[0], skip_special_tokens=True),
                }
                # print(result)
                examples.append(result)

        save_to_disk(
            examples, f"{save_path}/counterfactual_samples_{model_identifier}_{c}.jsonl"
        )
        # c += 1
