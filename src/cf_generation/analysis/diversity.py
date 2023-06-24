from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from munch import Munch
import itertools
from src.few_shot.utils import save_to_disk

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"


def compute_self_bleu(docs, base_doc, kwargs=None):
    # it should just be augments around one example.
    scores = []
    if len(docs) == 0:
        return Munch(bleu4=1)
    included = []
    data_points = []
    for doc in docs:
        included.append(doc)
        data_points.append([d for d in doc])
    # print("Included: ", included)
    # print("data points: ", data_points)
    included.append(base_doc)
    # print("Included: ", included)
    data_points.append([d for d in base_doc])
    # print("data points: ", data_points)

    points = list(itertools.combinations(range(len(included)), 2))
    # print(points)
    for i, j in points:
        scores.append(sentence_bleu([data_points[i]], data_points[j] ))
    return Munch(bleu4=np.mean(scores))


if __name__ == '__main__':
    # docs = ["How are you?"]
    # base_doc = "How are you?"
    # div_score = compute_self_bleu(docs, base_doc)
    # print(div_score)

    from datasets import load_dataset
    from tqdm import tqdm
    import jsonlines
    import statistics

    # load squad data
    dataset = load_dataset("squad", "plain_text")
    train_data = dataset["train"]
    squad_data = [sample for sample in tqdm(
        train_data, total=len(train_data), desc="Loading SQuAD data ... ")]

    self_blue = []
    examples_sim = []
    examples_div, examples_div1, examples_div2, examples_div3 = [], [], [], []
    c = 0
    with jsonlines.open(f"{BASE_PATH}src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_min_filtered_final_dedup_1.jsonl") as reader:
        for example in tqdm(reader):
            id = example["id"].split("_")[0]
            cf_question = example["question"]
            orig_example = [sample for sample in squad_data if sample["id"] == id][0]
            orig_question = orig_example["question"]
            bleu_score = compute_self_bleu([orig_question], cf_question)["bleu4"]
            self_blue.append(bleu_score)
            # print(self_blue)

            # c+=1
            # # if c == 10:
            # #     break
            # # if 0.1 <= bleu_score < 0.2:
            # #     examples_div.append(example)
            # if 0 <= bleu_score < 0.15:
            #     examples_div1.append(example)
            # elif 0.15 <= bleu_score < 0.3:
            #     examples_div2.append(example)
            # elif 0.3 <= bleu_score < 0.45:
            #     examples_div3.append(example)
            # else:
            #     examples_div.append(example)

        div_score = statistics.mean(self_blue)
        # save_to_disk(examples_sim, "counterfactual_data_flan_ul2_qg_pipeline_all_data_cleaned_sim_0.45.jsonl")
        # save_to_disk(examples_div, "counterfactual_data_flan_ul2_qg_pipeline_all_data_cleaned_div_0.1_0.2.jsonl")
        # save_to_disk(examples_div1, "counterfactual_data_flan_ul2_qg_pipeline_all_data_cleaned_div_0_0.15.jsonl")
        # save_to_disk(examples_div2, "counterfactual_data_flan_ul2_qg_pipeline_all_data_cleaned_div_0.15_0.3.jsonl")
        # save_to_disk(examples_div3, "counterfactual_data_flan_ul2_qg_pipeline_all_data_cleaned_div_0.3_0.45.jsonl")

    print(div_score)
