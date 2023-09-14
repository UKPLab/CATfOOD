import pandas as pd
import string
import re
from collections import Counter

import src.calibration.baseline.dataloader as dataloader


# BASE_PATH = "/home/sachdeva/projects/ukp/exp_calibration/"
BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

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
        score = metric_max_over_ground_truths(exact_match_score, pred_text, gold_text)
    elif metric == "f1":
        score = metric_max_over_ground_truths(f1_score, pred_text, gold_text)
    return score


def macro_ce(preds, golds, metric):
    """Macro CE"""
    ice_pos, ice_neg = 0, 0
    num_pos, num_neg = 0, 0
    for i in range(len(preds)):
        pred_text = preds[i]["text"]
        pred_proba = preds[i]["probability"]
        gold_text = golds[i]
        score = get_score(metric, pred_text, gold_text)
        if score == 1:
            ice_pos += 1 - pred_proba
            num_pos += 1
        else:
            ice_neg += pred_proba - 0
            num_neg += 1
    ice_pos /= num_pos
    ice_neg /= num_neg
    macro_ce = (ice_pos + ice_neg) / 2
    print("Macro CE: ", macro_ce)
    return macro_ce


if __name__ == "__main__":
    pred_df = (
        pd.read_json(
            BASE_PATH
            + "src/data/squad_adv_new/nbest_predictions_roberta_amb_0.75_top20.json"
        )
        .T.rename_axis("id")
        .reset_index()
    )
    answers = [f"answer_{i}" for i in range(20)]
    pred_df.columns = ["id"] + answers
    squad_data = dataloader.PreprocessData(
        "squad_adversarial", "AddSent", save_data=False, save_path="../../../../"
    )
    gold_text = [
        (sample["id"], sample["answers"]["text"])
        for sample in squad_data.processed_val_set()
    ]
    gold_df = pd.DataFrame(gold_text, columns=["id", "gold_answers"])
    data = pd.merge(pred_df, gold_df, on="id")
    pred_data = data.answer_0.values
    gold_data = data.gold_answers.values
    macro_ce(pred_data, gold_data, "exact_match")
