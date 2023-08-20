# evaluation metrics for the paper https://arxiv.org/pdf/2111.07367.pdf
import os
import torch

BASE_PATH="/storage/ukp/work/anon/research_projects/exp_calibration/"

def load_bin(fname):
    with open(fname, 'rb') as f:
        try:
            import pickle
            return pickle.load(f)
        except Exception as e:
            print(f"Error occurred: {e}")

def load_torch_data(fname):
    return torch.load(fname)


def top_k(data, k):
    """topk function"""
    tokens = data['feature'][0]
    attribution = data['attribution']
    values, indices = torch.topk(attribution, k)
    topk_tokens = [tokens[i] for i in indices]
    return topk_tokens

def ground_truth(data, k):
    """Ground truth"""
    ground_truth_tokens = ["<start>", "<end>"]
    tokens = data['feature'][0]
    attribution = data['attribution']
    values, indices = torch.topk(attribution, k)
    # check if the ground truth tokens are in the topk tokens and add to list
    topk_tokens = [tokens[i] for i in indices]
    topk_gt_tokens = [x for x in topk_tokens if x in ground_truth_tokens]
    return topk_gt_tokens


def precision_at_k(data_path, method, prefix, k=5):
    """Precision at k metric.
    measure over the top-k tokens
    in a salience ranking where k is the shortcut size.
    With s, m and xi denoting a salience method, a
    trained model m and the ith example from the synthetic
    set D and assuming two functions, topk(·)2
    and gtk(·) which output the top-k tokens from a
    salience ranking and the ground truth, respectively
    """
    fnames = os.listdir(os.path.join(data_path, method, prefix))
    fullnames = [os.path.join(data_path, method, prefix, x) for x in fnames]

    precision = 0
    sum = 0
    fullnames = fullnames[:1000]
    for i, file in enumerate(fullnames):
        data = load_torch_data(file)
        topk = top_k(data, k)
        gt = ground_truth(data, k)
        sum += len(set(topk).intersection(gt))
    precision = sum / (len(fullnames) * k)
    print(f"Precision at k={k} is {precision}")
    return precision


if __name__ == '__main__':
    path = f"{BASE_PATH}exp_roberta_with_shortcuts_0.3_v3/"
    method = "lime"
    prefix = "squad/dev/roberta/"
    precision_at_k(path, method, prefix, k=5)
