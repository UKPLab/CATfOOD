import jsonlines
from tqdm import tqdm

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"

def remove_duplicates():
    examples = []
    with jsonlines.open(BASE_PATH + "src/data/squad/squad_counterfactuals_noise_min_filtered.jsonl") as reader:
        for example in tqdm(reader):
            # get all samples for a particular id
            # idx = example["id"]
            # if "turk0" not in idx:
            #     continue
            examples.append(example)
    filtered = []
    for example in tqdm(examples, total=len(examples), desc="Removing duplicates ... "):
        question = example["question"]
        # remove duplicate questions from examples
        if question not in [e["question"] for e in filtered]:
            filtered.append(example)
    save_to_disk(filtered, BASE_PATH + "src/data/squad/squad_counterfactuals_noise_min_filtered_final.jsonl")
    # for sample in examples:
    #     original_idx = sample["id"].split("_")[0]
    #     original_question = example["question"]


def remove_duplicate_counterfactuals():
    """
    Keep only one example per id
    """
    examples = []
    with jsonlines.open(BASE_PATH + "src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_min_filtered_final_1.jsonl") as reader:
        for example in tqdm(reader):
            examples.append(example)
    filtered = []
    for example in tqdm(examples, total=len(examples), desc="Removing duplicate cfs ... "):
        idx = example["id"].split("_")[0]
        # remove duplicate questions from examples
        if idx not in [e["id"].split("_")[0] for e in filtered]:
            filtered.append(example)
    save_to_disk(filtered, BASE_PATH + "src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_min_filtered_final_dedup_1.jsonl")


def save_to_disk(data, file_name):
    with jsonlines.open(file_name, "a") as writer:
        for example in tqdm(data, total=len(data), desc="Saving samples ... "):
            writer.write(example)


if __name__ == '__main__':
    remove_duplicate_counterfactuals()