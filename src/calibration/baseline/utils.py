import os
import pickle
import json
from collections import OrderedDict
import torch
import pandas as pd

BASE_PATH="/storage/ukp/work/sachdeva/research_projects/exp_calibration/"
# BASE_PATH = "/home/sachdeva/projects/ukp/exp_calibration//"


def dump_to_bin(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_bin(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def read_json(fname):
    with open(fname, encoding='utf-8') as f:
        return json.load(f)

def dump_json(obj, fname, indent=None):
    with open(fname, 'w', encoding='utf-8') as f:
        return json.dump(obj, f, indent=indent)

def build_file_dict():
    # prefix = 'squad_sample-addsent_roberta-base'
    prefix = BASE_PATH + 'interpretations/shap/trivia_dev_roberta'
    fnames = os.listdir(prefix)
    qa_ids = [x.split('-',1)[1].split('.')[0] for x in fnames]
    fullnames = [os.path.join(prefix, x) for x in fnames]
    output = dict(zip(qa_ids, fullnames))
    # print(output)
    processed_instances = OrderedDict()
    for id in qa_ids:
        processed_instances[id] = torch.load(output[id])
        break
    print(processed_instances)
    # import torch
    # attr = torch.load()
    # return dict(zip(qa_ids, fullnames))


def lemmatize_pos_tag(x):
    """
    limit the amount of POS tags to reduce the number of features
    :param x:
    :return:
    """
    tok, pos, tag = x
    if tag == 'NNS':
        tag = 'NN'
    if tag == 'NNPS':
        tag = 'NNP'
    if tag.startswith('JJ'):
        tag = 'JJ'
    if tag.startswith('RB'):
        tag = 'RB'
    if tag.startswith('W'):
        tag = 'W'
    if tag.startswith('PRP'):
        tag = 'PRP'
    if tag.startswith('VB'):
        tag = 'VB'
    if pos == 'PUNCT':  # or tag in [":"]:
        tag = 'PUNCT'
    # if tag == '$':
    #     tag = 'SYM'
    # if tag == "``" or tag == "''":
    #     tag = "AUX"
    return tok, pos, tag
# trivia_tag_roberta_base_info

def convert_predictions_to_json_format():
    pred_df = pd.read_json(BASE_PATH+"src/data/squad/nbest_predictions_roberta_squad_easy_0.75_top20.json")\
        .T.rename_axis("id").reset_index()
    print(pred_df.head())
    answers = [f"answer_{i}" for i in range(20)]
    pred_df.columns = ["id"] + answers

    from src.calibration.baseline import dataloader
    from tqdm import tqdm
    data = dataloader.PreprocessData("squad_adversarial", "AddSent", save_data=False, save_path="../../../../")
    val_set = data.processed_val_set()
    gold_text = [(sample["id"], sample["answers"]["text"]) for sample in tqdm(val_set)]
    # gold_text = [(sample["id"], sample["answers"]) for sample in tqdm(trivia_data)]
    # print(gold_text)
    gold_df = pd.DataFrame(gold_text, columns=["id", "gold_answers"])
    data = pd.merge(pred_df, gold_df, on="id")
    # rename columns
    data = data.rename(columns={"gold_answers": "gold_text"})
    print(data.head())
    print(data.columns)
    print(data["answer_0"][0])


if __name__ == '__main__':
    # build_file_dict()
    # fname = "src/data/squad_adversarial/calib_cf_ig.bin"
    # x = load_bin(fname)
    # for i in x:
    #     print(i)
    # print(x[-1])
    # unq = []
    # p_unq = []
    # c = 0
    # for q_id in x:

    #     tags = x[q_id]
    #     words, tags_for_tok = tags['words'], tags['tags']
    #     if q_id == "56f879bdaef23719006260e2":
    #         print(tags)
    #     # print(tags_for_tok)
    #     tags_for_tok = [lemmatize_pos_tag(x) for x in tags_for_tok]# if x[2] not in ["``", "''", "SOS", "EOS"]]
    #     if q_id == "56f879bdaef23719006260e2":
    #         print(tags)
    #         print(tags_for_tok)
    #     p = [t[1] for t in tags_for_tok]
    #     t = [t[2] for t in tags_for_tok]
    #     for v in t:
    #         if v not in unq:
    #             unq.append(v)
    #     for v in p:
    #         if v not in p_unq:
    #             p_unq.append(v)
    #     # c += 1
    #     # if c == 1000:
    #     #     break
    # print(len(unq))
    # print(unq)
    #
    # print(len(p_unq))
    # print(p_unq)
    # base_path = "/home/sachdeva/projects/ukp/exp_calibration"
    # file = base_path + "/src/data/squad_adv_new/calibration_data.bin"
    # data = load_bin(file)
    # print(data)
    convert_predictions_to_json_format()
