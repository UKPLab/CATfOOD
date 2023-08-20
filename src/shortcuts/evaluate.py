from collections import Counter
import re
import string
import pandas as pd
from tqdm import tqdm
import argparse

import src.calibration.baseline.dataloader as dataloader
# from token_in_context import Shortcut


BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"
# BASE_PATH = "/home/anon/projects/ukp/exp_calibration/"

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    Compute f1 score of prediction
    :param prediction:
    :param ground_truth:
    :return:
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """
    Exact match for Squad evaluation
    :param prediction:
    :param ground_truth:
    :return:
    """
    # print("-------------------")
    # print(normalize_answer(prediction), normalize_answer(ground_truth))
    if normalize_answer(prediction) == normalize_answer(ground_truth):
        return 1
    else:
        return 0


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    # print(scores_for_ground_truths)
    return max(scores_for_ground_truths)


def get_score(metric, pred_text, gold_text):
    # print(pred_text, gold_text)
    # print(type(pred_text[0][0]["answer"]))
    if not isinstance(pred_text, str):
        pred_text = pred_text[0][0]["answer"]
    # print(gold_text, pred_text)
    if isinstance(gold_text, str):
        gold_text = [gold_text]
    if metric == "exact_match":
        score = metric_max_over_ground_truths(exact_match_score,
                                              pred_text,
                                              gold_text)
    elif metric == "f1":
        score = metric_max_over_ground_truths(f1_score, pred_text, gold_text)
    return score


def get_avg_scores(args):

    pred_df = pd.read_json(
        f"{BASE_PATH}src/data/{args.dataset}/nbest_predictions_{args.model_name}.json").T.rename_axis("id").reset_index()
    # print(pred_df.head())

    answers = [f"answer_{i}" for i in range(20)]
    pred_df.columns = ["id"] + answers
    if args.dataset == "squad":
        loader = dataloader.PreprocessData("squad", "plain_text", save_data=False, save_path="../../")
        data = loader.processed_val_set()
    elif args.dataset == "squad_adversarial":
        loader = dataloader.PreprocessData("squad_adversarial", "AddSent", save_data=False, save_path="../../")
        data = loader.processed_val_set()
    elif args.dataset == "trivia_qa":
        data = dataloader.get_dev_examples("./src/data", "dev_trivia.json")
    elif args.dataset == "hotpot_qa":
        data = dataloader.get_dev_examples("./src/data", "dev_hotpot.json")
    elif args.dataset == "news_qa":
        data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/NewsQA.jsonl")
    elif args.dataset == "bioasq":
        data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/BioASQ-dev.jsonl")
    elif args.dataset == "natural_questions":
        data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/NaturalQuestionsShort.jsonl")
    else:
        raise ValueError("Dataset not supported.")

    if args.dataset not in ["squad", "squad_adversarial"]:
        gold_text = [(sample["qas_id"], sample["answer_text"]) for sample in tqdm(data)]
    else:
        gold_text = [(sample["id"], sample["answers"]["text"]) for sample in tqdm(data)]

    # print(gold_text)
    gold_df = pd.DataFrame(gold_text, columns=["id", "gold_answers"])
    data = pd.merge(pred_df, gold_df, on="id")
    # print(data.head())
    scores_list = []
    metrics = ["exact_match", "f1"]
    for metric in metrics:
        scores = []
        for i in range(len(data)):
            scores.append(get_score(metric, data["answer_0"][i]["text"],
                                    data["gold_answers"][i]))
        # print(metric, sum(scores)/len(scores)*100)
        scores_list.append(sum(scores)/len(scores)*100)
    return scores_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Passing arguments for model, tokenizer, and dataset.")

    parser.add_argument(
        "--model_name", default="",
        type=str, required=True, help="Specify the model to use.")
    parser.add_argument("--dataset", type=str, required=True, help="Specify the dataset to use.")

    args = parser.parse_args()
    # print(pred_df["answer_0"])
    # dataloader = Shortcut(
    #     "squad",
    #     "plain_text",
    #     0.3,
    #     0.15,
    # )
    # train_set, val_set = dataloader.create_synthetic_set()
    scores_list = get_avg_scores(args)
    for score in scores_list:
        print(score)

