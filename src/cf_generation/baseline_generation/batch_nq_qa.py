"""
format:
question <sep_token> context

sep_token: ">>"

target: answer_text
"""

import os

import numpy as np
from tqdm import tqdm

import jsonlines

DOC_STRIDE = 320
MAX_LENGTH = 640
SEED = 42


def get_strided_contexts_and_ans(
        example, tokenizer, doc_stride=256, max_length=512, assertion=True
):
    # overlap will be of doc_stride - q_len

    question = example["question"]
    context = example["context"]
    answer = example["answer"]
    title = example["title"]
    answer_text = answer["text"]
    start_token = answer["start_token"]
    end_token = answer["end_token"]
    sep_token = ">>"

    input = f"{question} {sep_token} {context}"
    label = answer["text"]

    input_ids = tokenizer(input)["input_ids"]
    label_ids = tokenizer(label, max_length=256, padding=True, truncation=True)
    label_ids = np.array(label_ids["input_ids"])

    sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
    q_len = input_ids.index(sep_token_id) + 1
    q_indices = input_ids[:q_len]
    end_token_id = input_ids[-1]  # </s>
    # print(input_ids)

    # 1. return yes/no
    if answer["category"] in [3, 4]:  # category is an integer
        inputs = []
        category = []
        doc_start_indices = range(q_len, len(input_ids), max_length - doc_stride)
        for i in doc_start_indices:
            end_index = i + max_length - q_len
            slice = input_ids[i:end_index]
            inputs.append(q_indices + slice)
            if inputs[-1] != end_token:
                inputs.append(end_token)
            category.append(answer["category"])
            # if slice[-1] == tokenizer.sep_token_id:
            #     break

        return {
            "example_id": example["id"],
            "input_ids": inputs,
            "labels": [label_ids.tolist()] * len(category),
            "answers": {
                "start_token": [-100] * len(category),
                "end_token": [-100] * len(category),
                "category": category,
            },
        }

    # print(answer)
    splitted_context = example["context"].split(" ")
    complete_end_token = splitted_context[answer["end_token"]]
    # print(complete_end_token)
    answer["start_token"] = len(
        tokenizer(
            " ".join(splitted_context[: answer["start_token"]]),
            add_special_tokens=True,
        ).input_ids
    )
    answer["end_token"] = len(
        tokenizer(
            " ".join(splitted_context[: answer["end_token"]]),
            add_special_tokens=True
        ).input_ids
    )
    # print(answer)

    answer["start_token"] += q_len - 1  # include sep
    answer["end_token"] += q_len - 1

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

    # 2. when input length is less than our max_length
    if len(input_ids) <= max_length:
        return {
            "example_id": example["id"],
            "input_ids": [input_ids],
            "labels": [label_ids.tolist()],
            "answers": {
                "start_token": [answer["start_token"]],
                "end_token": [answer["end_token"]],
                "category": [answer["category"]],
            },
        }

    doc_start_indices = range(q_len, len(input_ids), max_length - doc_stride)
    # print(doc_start_indices)

    # 3. for strided qa samples
    inputs = []
    answers_start_token = []
    answers_end_token = []
    answers_category = []  # null, yes, no, long, short
    for i in doc_start_indices:
        end_index = i + max_length - q_len - 1
        slice = input_ids[i:end_index]
        if slice[-1] != end_token_id:
            slice.append(end_token_id)
        inputs.append(q_indices + slice)
        assert len(inputs[-1]) <= max_length, "Issue in truncating length"

        if start_token >= i and end_token <= end_index - 1:
            start_token = start_token
            end_token = end_token
            answers_category.append(answer["category"])  # ["short"] -> "short"
        else:
            start_token = -100
            end_token = -100
            answers_category.append("null")
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
            "category": answers_category,
        },
    }


def prepare_inputs(
        example, tokenizer, doc_stride=256, max_length=512, assertion=True
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
            cat = example["answers"]["category"]
            # print(start, end , cat)
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
                            "category": cat,
                        }
                    },
                )


if __name__ == "__main__":
    """Running area"""
    # from datasets import load_dataset
    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [">>"]}
    )
    # print(tokenizer.convert_tokens_to_ids(">>"))
    # model.resize_token_embeddings(len(tokenizer))
    c = 0
    data = []
    cache_file_name = "data/nq-train-tokenized-short-qa"
    with jsonlines.open("data/nq-train-short-only.jsonl") as file:
        for line in tqdm(file.iter()):
            # print("--------------------------")
            c += 1
            # if line["answer"]["category"] == 3:
            # print(line)
            example = prepare_inputs(line, tokenizer, max_length=MAX_LENGTH, doc_stride=DOC_STRIDE, assertion=False)
            # print(example)
            # if c == 10:
            #     break
            data.append(example)
            if c % 1000 == 0:
                save_to_disk(data, file_name=cache_file_name + ".jsonl")
                data = []
            if c % 5000 == 0:
                print(f"Samples processed: {c}")
    print(c)
