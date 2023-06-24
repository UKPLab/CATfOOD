import json
import os

import numpy as np
from tqdm import tqdm

SEED = 42
PROCESS_TRAIN = os.environ.pop("PROCESS_TRAIN", "false")
CATEGORY_MAPPING = {"null": 0, "short": 1, "long": 2, "yes": 3, "no": 4}


def _get_single_answer(example):
    def choose_first(answer, is_long_answer=False):
        # print(answer)
        if is_long_answer:
            assert isinstance(answer, dict)
            answer = [answer]
        else:
            assert isinstance(answer, list)
        if len(answer) == 1:
            answer = answer[0]
            return {k: [answer[k]][0] for k in answer} if is_long_answer else answer
        for a in answer:
            if is_long_answer:
                a = {k: [a[k]] for k in a}
            if len(answer) > 1:
                break
        return a
    # print(example)
    answer = {"id": example["example_id"]}
    annotation = example["annotations"]
    # print(annotation)
    if isinstance(annotation, list):
        annotation = annotation[0]
    yes_no_answer = annotation["yes_no_answer"]
    if yes_no_answer != "NONE":
        answer["category"] = ["yes"] if yes_no_answer == "YES" else ["no"]
        answer["start_token"] = answer["end_token"] = []
        answer["start_byte"] = answer["end_byte"] = []
        answer["text"] = ["<cls>"]
    elif annotation["short_answers"]:
        answer["category"] = ["short"]
        # print(annotation["short_answers"])
        out = choose_first(annotation["short_answers"])
        out["text"] = []
        answer.update(out)
        # print(out["start_token"])
    else:
        # answer will be long if short is not available
        answer["category"] = ["long"]
        out = choose_first(annotation["long_answer"], is_long_answer=True)
        out["text"] = []
        answer.update(out)
    # print(answer)

    # disregard some samples
    if answer["start_token"] == answer["end_token"]:
        answer["remove_it"] = True
    else:
        answer["remove_it"] = False

    # cols = ["start_token", "end_token", "start_byte", "end_byte", "text"]
    # if not all([isinstance(answer[k], list or int) for k in cols]):
    #     raise ValueError("Issue in ID", example["example_id"])

    return answer


def get_context_and_ans(example, assertion=False):
    """Gives new context after removing <html> & new answer tokens as per new context"""
    answer = _get_single_answer(example)
    # bytes are of no use
    del answer["start_byte"]
    del answer["end_byte"]

    # handle yes_no answers explicitly
    if answer["category"][0] in ["yes", "no"]:  # category is list with one element
        doc = example["document_tokens"]
        context = []
        for i in range(len(doc)):
            if not doc[i]["html_token"]:
                context.append(doc[i]["token"])
        return {
            "context": " ".join(context),
            "answer": {
                "start_token": -100,  # ignore index in cross-entropy
                "end_token": -100,  # ignore index in cross-entropy
                "category": answer["category"],
                "span": answer["category"],  # extra
            },
        }

    # later, help in removing all no answers
    if answer["start_token"] == [-1]:
        return {
            "context": "None",
            "answer": {
                "start_token": -1,
                "end_token": -1,
                "category": "null",
                "span": "None",  # extra
            },
        }

    # handling normal samples
    # cols = ["start_token", "end_token"]
    # answer.update(
    #     {k: answer[k][0] if len(answer[k]) > 0 else answer[k] for k in cols}
    # )  # e.g. [10] == 10

    doc = example["document_tokens"]
    # if len(doc) > 1:
    #     doc = doc[0]
    start_token = answer["start_token"]
    end_token = answer["end_token"]

    context = []
    for i in range(len(doc)):
        if not doc[i]["html_token"]:
            context.append(doc[i]["token"])
        else:
            if answer["start_token"] > i:
                start_token -= 1
            if answer["end_token"] > i:
                end_token -= 1
    new = " ".join(context[start_token:end_token])
    # print(new)

    # checking above code
    if assertion:
        """checking if above code is working as expected for all the samples"""
        is_html = doc["html_token"][answer["start_token"] : answer["end_token"]]
        old = doc["token"][answer["start_token"]: answer["end_token"]]
        old = " ".join([old[i] for i in range(len(old)) if not is_html[i]])
        if new != old:
            print("ID:", example["id"])
            print("New:", new, end="\n")
            print("Old:", old, end="\n\n")

    return {
        "context": " ".join(context),
        "answer": {
            "start_token": start_token,
            "end_token": end_token - 1,  # this makes it inclusive
            "category": answer["category"],  # either long or short
            "span": new,  # extra
        },
    }


def get_contexts_and_ans(
    example, assertion=True
):
    # overlap will be of doc_stride - q_len
    # print(example)
    out = get_context_and_ans(example, assertion=assertion)
    out["question"] = example["question_text"]
    out["example_id"] = example["example_id"]
    out["document_title"] = example["document_title"]
    answer = out["answer"]

    # later, removing these samples
    if answer["start_token"] == -1:
        return {
            "example_id": example["example_id"],
            "title": example["document_title"],
            "question": example["question_text"],
            "context": "",
            "answer": {
                "start_token": -1,
                "end_token": -1,
                "category": ["null"],
                "span": ""
            },
        }

    return out


def prepare_inputs(
    example, assertion=False
):
    example = get_contexts_and_ans(
        example,
        assertion=assertion,
    )

    return example


def save_to_disk(hf_data, file_name):
    with jsonlines.open(file_name, "a") as writer:
        for example in tqdm(hf_data, total=len(hf_data), desc="Saving samples ... "):
            answer = example["answer"]
            if answer["start_token"] == -1 and answer["end_token"] == -1:
                continue  # leave waste samples with no answer
            if answer["category"][0] == "null" and np.random.rand() < 0.6:
                continue  # removing 50 % samples
            writer.write(
                {
                    "id": example["example_id"],
                    "title": example["document_title"],
                    "question": example["question"] + '?',
                    "context": example["context"],
                    "answer": {
                        "start_token": answer["start_token"],
                        "end_token": answer["end_token"],
                        "category": CATEGORY_MAPPING[answer["category"][0]],
                        "text": answer["span"]
                    }
                }
            )


if __name__ == "__main__":
    """Running area"""
    from pathlib import Path
    import jsonlines
    import os
    import gzip

    path = Path("/home/anon/Downloads/v1.0/dev")
    files = os.listdir(path)
    dataset_files = [path / str(f) for f in files]
    # dataset_file = dataset_files[0]

    c = 0
    data = []
    cache_file_name = "data/nq-dev-short-only"
    for data_file in tqdm(dataset_files):
        with gzip.open(data_file, "rb") as f:
            for line in f:
                # print(json.loads(line)["annotations"])
                # print(len(json.loads(line)["annotations"]))
                if json.loads(line)["annotations"][0]["short_answers"]:
                #     print(json.loads(line)["annotations"][0]["short_answers"])
                #     print(json.loads(line)["annotations"][0]["long_answer"])
                # print("--------------------------")
                # c += 1
                # if c == 100:
                #     break
                    example = prepare_inputs(json.loads(line))
                    data.append(example)
                    c += 1
                if c % 1000 == 0:
                    save_to_disk(data, file_name=cache_file_name + ".jsonl")
                    data = []
                if c % 5000 == 0:
                    print(f"Samples processed: {c}")
            # if c == 20:
            #     break
            # break

    # print(example)
    # save_to_disk(data, file_name=cache_file_name + ".jsonl")
    # with jsonlines.open("/home/anon/Downloads/v1.0/train/data.jsonl") as file:
    #     for line in file.iter():
    #         # print(line)
    #         # print("--------------------------")
    #         c += 1
    #         # prepare_inputs(line, tokenizer)
    #         # if c == 10:
    #         #     break
    # print(c)
