import os

import numpy as np
from tqdm import tqdm

from datasets import load_dataset
from transformers import T5Tokenizer

import jsonlines

DOC_STRIDE = 320
MAX_LENGTH = 640
SEED = 42

BASE_PATH = "/storage/xyz/work/anon/research_projects/exp_calibration/"

def _add_eos_examples(example):
    pass


def get_strided_contexts_and_ans(
        example, tokenizer, doc_stride=256, max_length=512, assertion=True
):
    # overlap will be of doc_stride - q_len

    context = example["context"]
    answer = example["answers"]
    answer_text = answer["text"][0]
    start_token = answer["answer_start"][0]
    end_token = start_token + len(answer_text)
    hl_token = "<hl>"
    # sep_token = "<sep>"

    # for short/long questions
    input = f"generate question: {context[:start_token]} {hl_token} {answer_text} {hl_token} {context[end_token + 1:]}"
    label = example["question"]
    input_ids = tokenizer(input)["input_ids"]
    label_ids = tokenizer(label, max_length=256, padding=True, truncation=True)
    label_ids = np.array(label_ids["input_ids"])

    label_ids[label_ids == tokenizer.pad_token_id] = -100

    cmd_indices = input_ids[:3]
    cmd_len = len(cmd_indices)
    end_token_id = input_ids[-1]  # </s>

    splitted_context = example["context"].split(" ")
    complete_end_token = splitted_context[end_token]
    # print(complete_end_token)
    answer["start_token"] = len(
        tokenizer(
            " ".join(splitted_context[: start_token]),
            add_special_tokens=True,
        ).input_ids
    )
    answer["end_token"] = len(
        tokenizer(
            " ".join(splitted_context[: end_token]),
            add_special_tokens=True
        ).input_ids
    )
    # print(answer)

    answer["start_token"] += cmd_len + 1  # include highlight token
    answer["end_token"] += cmd_len + 1  # include highlight token

    # fixing end token
    num_sub_tokens = len(
        tokenizer(complete_end_token,
                  add_special_tokens=False).input_ids
    )
    if num_sub_tokens > 1:
        answer["end_token"] += num_sub_tokens - 1

    old = input_ids[
        answer["start_token"]: answer["end_token"] + 1
    ]  # right & left are inclusive
    start_token = answer["start_token"]
    end_token = answer["end_token"]

    # print(start_token, end_token)

    if assertion:
        """This won't match exactly because of extra gaps => visually inspect everything"""
        new = tokenizer.decode(old)
        if answer_text != new:
            print("ISSUE IN TOKENIZATION")
            print("OLD:", answer_text)
            print("NEW:", new, end="\n\n")

    if len(input_ids) <= max_length:
        return {
            "example_id": example["id"],
            "input_ids": [input_ids],
            "labels": [label_ids.tolist()],
            "answers": {
                "start_token": [answer["start_token"]],
                "end_token": [answer["end_token"]],
            },
        }

    doc_start_indices = range(cmd_len, len(input_ids), max_length - doc_stride)
    # print(doc_start_indices)

    inputs = []
    answers_start_token = []
    answers_end_token = []
    for i in doc_start_indices:
        end_index = i + max_length - cmd_len - 1
        slice = input_ids[i:end_index]
        if slice[-1] != end_token_id:
            slice.append(end_token_id)
        inputs.append(cmd_indices + slice)
        assert len(inputs[-1]) <= max_length, "Issue in truncating length"

        if start_token >= i and end_token <= end_index - 1:
            start_token = start_token
            end_token = end_token
        else:
            start_token = -100
            end_token = -100
        # print(start_token, end_token)
        new = inputs[-1][start_token: end_token + 1]

        answers_start_token.append(start_token)
        answers_end_token.append(end_token)
        if assertion:
            """checking if above code is working as expected for all the samples"""
            if new != old and new != [tokenizer.cls_token_id]:
                print("ISSUE in strided for ID:", example["id"])
                print("New:", tokenizer.decode(new))
                print("Old:", tokenizer.decode(old), end="\n\n")

    return {
        "example_id": example["id"],
        "input_ids": inputs,
        "labels": [label_ids.tolist()] * len(inputs),
        "answers": {
            "start_token": answers_start_token,
            "end_token": answers_end_token,
        },
    }


def prepare_inputs(
        example, tokenizer, doc_stride=256, max_length=512, assertion=False
):
    example = get_strided_contexts_and_ans(
        example,
        tokenizer,
        doc_stride=doc_stride,
        max_length=max_length,
        assertion=assertion,
    )

    return example


def save_to_disk(hf_data, file_name):
    with jsonlines.open(file_name, "a") as writer:
        for example in tqdm(hf_data, total=len(hf_data), desc="Saving samples ... "):
            start = example["answers"]["start_token"]
            end = example["answers"]["end_token"]
            for input_ids, labels, start, end, cat in zip(
                    example["input_ids"],
                    example["labels"],
                    start,
                    end,
                    cat
            ):
                if start == -100 and end == -100:
                    continue  # remove unanswerable questions

                writer.write(
                    {
                        "id": example["example_id"],
                        "input_ids": input_ids,
                        "labels": labels,
                        "answers": {
                            "start_token": start,
                            "end_token": end,
                        }
                    },
                )



if __name__ == "__main__":
    """Running area"""


    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<hl>"]}
    )
    # model.resize_token_embeddings(len(tokenizer))
    c = 0
    data = []
    cache_file_name = os.path.join(BASE_PATH, "src/data/squad/squad_qg_tokenized")

    # # load squad data
    dataset = load_dataset("squad", "plain_text")
    train_data = dataset["train"]
    squad_data = [sample for sample in tqdm(
        train_data, total=len(train_data), desc="Loading SQuAD data ... ")]
    #
    for ex in tqdm(train_data):
        example = prepare_inputs(ex, tokenizer, max_length=MAX_LENGTH, doc_stride=DOC_STRIDE)
        data.append(example)
        if c % 1000 == 0:
            save_to_disk(data, file_name=cache_file_name + ".jsonl")
            data = []
        if c % 5000 == 0:
            print(f"Samples processed: {c}")
    print(c)
