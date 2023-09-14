from nltk.translate.bleu_score import sentence_bleu
from zss import simple_distance, Node
import numpy as np
import edit_distance
from munch import Munch
import itertools
import random

random.seed(42)

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"


def normalized_levenshtein_distance(s1, s2):
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(
                        1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                    )
            distances = distances_
        return distances[-1]

    distance = levenshtein_distance(s1, s2)
    max_length = max(len(s1), len(s2))
    return distance / max_length


def compute_lev_distance(doc1, doc2):
    sm = edit_distance.SequenceMatcher(a=[t for t in doc1], b=[t for t in doc2])
    return 1 - sm.ratio()


def add_node_to_root(root):
    if not root or not list(root.children):
        return None
    curr_node = Node(root.text.lower())
    for c in root.children:
        cnode = add_node_to_root(c)
        if cnode:
            curr_node.addkid(cnode)
    return curr_node


def get_nodes(doc):
    span = list(doc.sents)[0]
    return add_node_to_root(span.root)


def compute_tree_edit_distance(doc1, doc2):
    try:
        dist = simple_distance(get_nodes(doc1), get_nodes(doc2))
        return dist  # / len(doc1)
    except:
        return 0


def compute_closeness(docs, base_doc, sentence_similarity=None):
    sem_dist, tree_dist, edit_dist = [], [], []
    for doc in docs:
        if sentence_similarity:
            sem_dist.append(sentence_similarity(base_doc, doc))
        tree_dist.append(compute_tree_edit_distance(base_doc, doc))
        edit_dist.append(compute_lev_distance(base_doc, doc))
    return Munch(
        sem_dist=np.mean(sem_dist),
        tree_dist=np.mean(tree_dist),
        edit_dist=np.mean(edit_dist),
    )


if __name__ == "__main__":
    # rag_counterfactuals_complete_noise_min_filtered_final_dedup_1
    # counterfactual_data_llama_13b_v1_qg_pipeline_all_data_cleaned.jsonl
    # flan_ul2_collated_data_with_answers_processed.jsonl
    # flan-t5-xxl-v3_collated_data_with_answers_processed.jsonl
    # counterfactual_data_alpaca_13b_v2_qg_pipeline_all_data_cleaned.jsonl
    # counterfactual_data_gpt_jt_v2_qg_pipeline_all_data_cleaned.jsonl
    # counterfactual_data_gpt_neox_20b_v2_qg_pipeline_all_data_cleaned.jsonl

    from datasets import load_dataset
    from tqdm import tqdm
    import jsonlines
    import statistics

    # load squad data
    dataset = load_dataset("squad", "plain_text")
    train_data = dataset["train"]
    squad_data = [
        sample
        for sample in tqdm(
            train_data, total=len(train_data), desc="Loading SQuAD data ... "
        )
    ]

    # for sample in squad_data:
    #     if sample["id"] == "570c4f18fed7b91900d45893":
    #         print(sample)

    levenshtein_dist = []
    c = 0
    with jsonlines.open(
        f"{BASE_PATH}src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_min_filtered_final_dedup_1.jsonl"
    ) as reader:
        for example in tqdm(reader):
            id = example["id"].split("_")[0]
            cf_question = example["question"]
            orig_example = [sample for sample in squad_data if sample["id"] == id][0]
            orig_question = orig_example["question"]
            # print("original: ", orig_question)
            # print("cf: ", cf_question)
            levenshtein_dist.append(
                normalized_levenshtein_distance(orig_question, cf_question)
            )
            # print(levenshtein_dist)
        #
        # c += 1
        # # break
        # if c == 20:
        #     break
        dist_score = statistics.mean(levenshtein_dist)
    print(dist_score)

    # x = levenshtein_distance("cap", "cap")
    # print(x)
