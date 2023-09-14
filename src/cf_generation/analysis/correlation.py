import numpy as np
from scipy import stats
from typing import List


def calculate_correlation(x: List, y: List):
    arr_a = np.array(x)
    arr_b = np.array(y)

    res = stats.spearmanr(arr_a, arr_b)
    corr, p_value = res
    return corr, p_value


if __name__ == "__main__":
    # sbert_sim  = [0.55, 0.48, 0.50, 0.50, 0.45, 0.41, 0.41]
    # semantic_eq = [0.52, 0.46, 0.51, 0.55, 0.46, 0.41, 0.40]
    # leven_dist = [0.61, 0.67, 0.65, 0.67, 0.68, 0.71, 0.71]
    # self_bleu = [0.30, 0.26, 0.28, 0.27, 0.24, 0.19, 0.19]

    sbert_sim = [0.55, 0.50, 0.45, 0.41]
    semantic_eq = [0.52, 0.51, 0.46, 0.40]
    leven_dist = [0.61, 0.65, 0.68, 0.71]
    self_bleu = [0.30, 0.28, 0.24, 0.19]

    ood_gain = [1.24, 3.02, 3.45, 1.27, 3.61, 2.36, 3.48]
    calib_conf_gain = [5.42, 7.87, 8.02, 8.18]
    calib_shap_gain = [1.53, 2.73, 2.62, 3.35]
    calib_sc_attn_gain = [1.57, 2.55, 2.6, 2.97]
    calib_ig_gain = [1.2, 2.75, 2.45, 3.35]

    calib_shap_gain_f = [1.53, 3.35, 3.13, 3.9]
    calib_sc_attn_gain_f = [1.57, 3.72, 3.63, 4.33]
    calib_ig_gain_f = [1.2, 3.83, 3.48, 4.42]

    bleu_corr = -calculate_correlation(self_bleu, calib_ig_gain_f)[0]
    leven_corr = calculate_correlation(leven_dist, calib_ig_gain_f)[0]  # label flipping
    sbert_corr = -calculate_correlation(sbert_sim, calib_ig_gain_f)[0]
    semantic_corr = -calculate_correlation(semantic_eq, calib_ig_gain_f)[0]

    avg_corr = (bleu_corr + leven_corr + sbert_corr + semantic_corr) / 4
    print(round(avg_corr, 2))

    print("self bleu: ", bleu_corr)
    print("levensh dist: ", leven_corr)
    print("sbert sim: ", sbert_corr)
    print("semantic eq: ", semantic_corr)
