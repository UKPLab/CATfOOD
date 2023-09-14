import jsonlines
from tqdm import tqdm
from datasets import load_dataset

# BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"
BASE_PATH = "//"

with jsonlines.open(
    f"{BASE_PATH}src/data/squad/squad_counterfactuals_noise_min_filtered_final_2.jsonl"
) as reader:
    filtered_data = [
        sample for sample in tqdm(reader, desc="Loading Counterfactual data ... ")
    ]
    filtered_ids = [data["id"].split("_")[0] for data in filtered_data]
    squad_dataset = load_dataset("squad", "plain_text")
    squad_examples = set()
    # print(len(data))
    print(len(filtered_ids))
    squad_ids = [data["id"] for data in squad_dataset["train"]]
    # check if the id is in the squad dataset and list the ones that are not
    extra_ids = [idx for idx in squad_ids if idx not in filtered_ids]

with jsonlines.open(
    f"{BASE_PATH}src/data/squad/squad_counterfactuals_llms.jsonl", "a"
) as writer:
    with jsonlines.open(
        f"{BASE_PATH}src/data/squad/squad_counterfactuals_noise_min_filtered.jsonl"
    ) as reader:
        for sample in tqdm(reader, desc="Loading Counterfactual data ... "):
            if sample["id"].split("_")[0] in extra_ids:
                writer.write(sample)
