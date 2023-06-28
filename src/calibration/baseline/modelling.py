import os
import sys
from collections import OrderedDict
from itertools import islice

sys.path.append('../..')
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from common import IndexedFeature, FeatureVocab
from utils import load_bin, dump_to_bin
from itertools import chain
import matplotlib.pyplot as plt
import collections

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from sklearn.metrics import roc_curve, auc, roc_auc_score
# from calib_exp.calib_metrics import f1auc_score
from sklearn.inspection import permutation_importance
import xgboost as xg
from sklearn.feature_selection import RFECV
import json
import shap


def _f1auc_score(x, y):
    x = np.ravel(x)
    y = np.ravel(y)
    desc_score_indices = np.argsort(x, kind="mergesort")[::-1]
    x = x[desc_score_indices]
    y = y[desc_score_indices]

    distinct_value_indices = np.where(np.diff(x, append=0))[0]
    # threshold_idxs = np.r_[distinct_value_indices, y.size - 1]

    # accumulate the true positives with decreasing threshold
    threshold_values = x[distinct_value_indices]

    # for t in threshold_values:
    #     results = np.array([np.mean(y[x < t])  for t in T])
    threshold_f1 = np.array([np.mean(y[:(t + 1)]) for t in distinct_value_indices])
    # print(threshold_values)
    # print(threshold_f1)
    return auc(threshold_values, threshold_f1)

def f1auc_score(score, f1):
    score = np.ravel(score)
    f1 = np.ravel(f1)
    sorted_idx = np.argsort(-score)
    score = score[sorted_idx]
    f1 = f1[sorted_idx]
    num_test = f1.size
    segment = min(1000, score.size - 1)
    T = np.arange(segment) + 1
    T = T/segment
    results = np.array([np.mean(f1[:int(num_test * t)])  for t in T])
    # print(results)
    return np.mean(results)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--method', type=str, default='shap')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--do_baseline', default=False, action='store_true')
    parser.add_argument('--do_maxprob', default=False, action='store_true')
    parser.add_argument('--do_bow', default=False, action='store_true')
    parser.add_argument('--rm_bow', default=False, action='store_true')
    parser.add_argument('--rm_link', default=False, action='store_true')
    parser.add_argument('--rm_baseline', default=False, action='store_true')
    parser.add_argument('--rm_func', default=False, action='store_true')
    parser.add_argument('--do_unnorm', default=False, action='store_true')
    parser.add_argument('--n_run', type=int, default=20)
    parser.add_argument('--train_size', type=int, default=500)
    parser.add_argument('--force_dev_size', type=int, default=0)
    parser.add_argument('--model', type=str, default='rf')
    parser.add_argument('--show_imp', default=False, action='store_true')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--arg_n_tree', type=int, default=100)
    parser.add_argument('--arg_max_depth', type=int, default=None)
    parser.add_argument('--quant_size', type=int, default=0)
    parser.add_argument('--quant_type', type=str, default='val')
    parser.add_argument('--include_cf', default=False, action='store_true')
    args = parser.parse_args()
    if args.split == None:
        args.split = 'addsent-dev' if args.dataset == 'squad' else 'dev'
    return args


def f1_prob_curve(f1, score):
    sorted_idx = np.argsort(-score)
    score = score[sorted_idx]
    f1 = f1[sorted_idx]
    num_test = f1.size
    # T = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    T = [0.25, 0.5, 0.75, 1.0]
    results = np.array([np.mean(f1[:int(num_test * t)]) for t in T])
    return results


def train_max_accuracy(x, y):
    x = x.flatten()
    best_acc = 0
    best_v = 0
    # print("Training max accuracy...")
    # print("x: ", x.shape, " y: ", y.shape)
    # print("x: ", x, " y: ", y)
    for v in x:
        p = x > v
        ac = np.sum(p == y) / y.size
        if ac > best_acc:
            best_acc = ac
            best_v = v
    # print("Best threshold: ", best_v, " acc: ", best_acc)
    return best_acc, best_v


def test_max_accuracy(x, y, v):
    x = x.flatten()
    p = x > v
    ac = np.sum(p == y) / y.size
    # convert true false to 1 0 in p
    preds = p.astype(int)
    # print("p", preds)
    return ac, preds


def feat_to_list(indexed_feat, vocab):
    val_feat = [.0] * len(vocab)
    for f, v in indexed_feat.data.items():
        val_feat[vocab[f]] = v
    return val_feat


def make_np_dataset(indexed_data):
    vocab = FeatureVocab()
    for k, v in indexed_data.items():
        feat = v['feature']
        for f in feat.data:
            # if f not in ["BOW_Q_XX", "BOW_A_XX"]:   #, "POS_NORMED_TOK_A_XX", "POS_NORMED_TOK_Q_XX"]:
            vocab.add(f)
    print('Total Num of Feature', len(vocab))
    print(vocab.get_names())
    y = np.array([v['label'] for v in indexed_data.values()])
    f1 = np.array([v['f1_score'] for v in indexed_data.values()])
    x = np.array([feat_to_list(v['feature'], vocab) for v in indexed_data.values()])

    for index, name in vocab.id_to_feat.items():
        if 'LENGTH' in name:
            x[:, index] = x[:, index] / np.max(x[:, index])
    return x, y, f1, vocab


def train_test_split(X, Y, F1, permuted_idx=None, ratio=0.75):
    num_data = X.shape[0]
    perm_idx = np.random.permutation(num_data) if permuted_idx is None else permuted_idx
    num_train = int(num_data * ratio)
    train_idx = perm_idx[:num_train]
    dev_idx = perm_idx[num_train:]
    train_x, train_y, train_F1 = X[train_idx, :], Y[train_idx], F1[train_idx]
    dev_x, dev_y, dev_F1 = X[dev_idx, :], Y[dev_idx], F1[dev_idx]

    return train_x, train_y, dev_x, dev_y, train_F1, dev_F1


def apply_train_test_split(X, Y, F1, train_test_split, force_dev_size=0):
    train_idx, dev_idx = train_test_split

    # for hyper par search
    if force_dev_size:
        train_size = len(train_idx)
        dev_idx = train_idx[(train_size - force_dev_size):]
        train_idx = train_idx[:(train_size - force_dev_size)]
    train_x, train_y, train_F1 = X[train_idx, :], Y[train_idx], F1[train_idx]
    dev_x, dev_y, dev_F1 = X[dev_idx, :], Y[dev_idx], F1[dev_idx]

    return train_x, train_y, dev_x, dev_y, train_F1, dev_F1


def interp_calibrator_model(cls, vocab):
    if isinstance(cls, LogisticRegression):
        feat_imp = cls.coef_.flatten()
    elif isinstance(cls, RandomForestClassifier):
        feat_imp = cls.feature_importances_
    elif isinstance(cls, GradientBoostingClassifier):
        feat_imp = cls.feature_importances_
    else:
        print('UnInterpretable model')
        return

    imp_idx = np.argsort(-np.abs(feat_imp))
    for i in imp_idx[:100]:
        print(vocab.get_word(i), feat_imp[i])


def proc_input_data(args, data):
    if args.dataset == 'squad':
        data = OrderedDict([(k, v) for (k, v) in data.items() if '-' in k])
    new_data = OrderedDict()
    # print('Total Num of Data', len(data))
    # print(data)
    for qas_id, ex in data.items():
        new_feat = IndexedFeature()
        for f, val in ex['feature'].data.items():
            if args.do_unnorm and 'NORMED' in f:
                continue
            if not args.do_unnorm and 'UNNORM' in f:
                continue
            # if f == 'FIRST_DISTINCT_PROB':
            #     continue
            # if f.startswith('BASELINE_PROB') and not f.startswith('BASELINE_PROB_0'):
            #     continue
            # if 'TOK_C_' in f:
            #     continue
            if 'TOK_IN_' in f:
                continue
            if 'BOW_IN_' in f:
                continue
            if not args.do_bow:
                if 'BOW_C' in f:
                    continue
            if f.startswith('ENT'):
                continue
            new_feat.add(f, val)

        new_data[qas_id] = {'label': ex['label'], 'f1_score': ex['f1_score'], 'feature': new_feat}
    data = new_data
    if args.rm_baseline:
        new_data = OrderedDict()
        for qas_id, ex in data.items():
            new_feat = IndexedFeature()
            for f, val in ex['feature'].data.items():
                if f.startswith('BASELINE') and not f.startswith('BASELINE_PROB_0'):
                    continue
                new_feat.add(f, val)
            new_data[qas_id] = {'label': ex['label'], 'f1_score': ex['f1_score'], 'feature': new_feat}
        data = new_data
    if args.do_baseline:
        new_data = OrderedDict()
        for qas_id, ex in data.items():
            new_feat = IndexedFeature()
            for f, val in ex['feature'].data.items():
                if not f.startswith('BASELINE' or 'REPR'):
                    continue
                # if not ('BASELINE' in f or 'REPR' in f):
                #     continue
                new_feat.add(f, val)
            new_data[qas_id] = {'label': ex['label'], 'f1_score': ex['f1_score'], 'feature': new_feat}
        data = new_data
    if args.do_maxprob:
        new_data = OrderedDict()
        for qas_id, ex in data.items():
            new_feat = IndexedFeature()
            for f, val in ex['feature'].data.items():
                if not f.startswith('BASELINE_PROB_0'):
                    continue
                # if not 'REPR' in f:
                #     continue
                new_feat.add(f, val)
            new_data[qas_id] = {'label': ex['label'], 'f1_score': ex['f1_score'], 'feature': new_feat}
        data = new_data
    if args.do_bow:
        new_data = OrderedDict()
        for qas_id, ex in data.items():
            new_feat = IndexedFeature()
            for f, val in ex['feature'].data.items():
                if not ('BASELINE' in f or 'BOW' in f):
                    continue
                # if not ('BASELINE' in f
                #         or f.startswith('BOW')
                        # or 'ENT' in f
                        # or 'WOR' in f
                        # or 'BOW_C_NNP' in f
                        # or 'WOR_at_SbL1C_C_C' in f
                        # or 'WOR_at_SbSBC_C_C' in f
                        # or 'WOR_at_SbCDC_C_C' in f
                        # or 'WOR_at_SbL1W_C_C' in f

                        # or 'TTR_CorrTTR_S_C' in f
                        # or 'TTR_SimpTTR_S_C' in f  # 64.1
                        # or 'TTR_BiLoTTR_S_C' in f  #f1 82.9
                        # or 'TTR_UberTTR_S_C' in f
                        # or 'TTR_CorrTTR_S_C' in f
                        # or 'VAR_SquaAjV_S_C' in f
                        # or 'VAR_at_CoTag_C_C' in f
                        # or 'VAR_CorrAjV_S_C' in f
                        # or 'VAR_SquaNoV_S_C' in f  # 64 and 83 f1
                        # or 'VAR_CorrNoV_S_C' in f  # gg
                        # or 'VAR_SquaVeV_S_C' in f  # gg 64.2
                        # or 'VAR_CorrVeV_S_C' in f   # gg 64.2
                        # or f.startswith('_POS')
                        # or '_POS_as_NoTag_C_Q' in f
                        # or '_POS_at_NoTag_C_C' in f
                        # or '_POS_at_NoTag_C_Q' in f
                        # or '_POS_to_NoTag_C_Q' in f
                        # or 'PSY_at_AAKuL_C_C' in f
                        # or 'PSY_at_AABiL_C_Q' in f
                        # or 'ENT_at_UEnti_C' in f  # use this
                        # or 'ENT_at_EntiM_C' in f        # use this
                        # or 'SHA_at_Sylla_C_Q' in f
                        # or 'SHA_as_Sylla_C_C' in f
                        # or 'SHA_at_Sylla_C_C' in f
                        # or 'SHA_TokSenL_S_C' in f
                # ):  # and 'BOW_A' in f:
                #     continue
                # if not ('SHA_at_Sylla_C_C' in f or 'ENT_at_UEnti_C' in f
                #         or 'SHA_at_Sylla_C_Q' in f or 'ENT_at_EntiM_C' in f or 'SHA_as_Sylla_C_C' in f
                #         or 'SHA_TokSenL_S_C' in f):
                #     continue
                # if 'BOW_A' in f:
                #     continue
                new_feat.add(f, val)
            new_data[qas_id] = {'label': ex['label'], 'f1_score': ex['f1_score'], 'feature': new_feat}
        data = new_data
    if args.rm_bow:
        new_data = OrderedDict()
        for qas_id, ex in data.items():
            new_feat = IndexedFeature()
            for f, val in ex['feature'].data.items():
                if 'BOW' in f:
                    continue
                new_feat.add(f, val)
            new_data[qas_id] = {'label': ex['label'], 'f1_score': ex['f1_score'], 'feature': new_feat}
        data = new_data
    if args.rm_link:
        new_data = OrderedDict()
        for qas_id, ex in data.items():
            new_feat = IndexedFeature()
            for f, val in ex['feature'].data.items():
                if 'LINK' in f and 'AGG' not in f:
                    continue
                new_feat.add(f, val)
            new_data[qas_id] = {'label': ex['label'], 'f1_score': ex['f1_score'], 'feature': new_feat}
        data = new_data
    # print(data)
    return data


def  get_feature_importances(cls):
    if isinstance(cls, LogisticRegression):
        feat_imp = cls.coef_.flatten()
    elif isinstance(cls, RandomForestClassifier):
        feat_imp = cls.feature_importances_
    elif isinstance(cls, GradientBoostingClassifier):
        feat_imp = cls.feature_importances_
    else:
        return None
    return feat_imp


def selection_based_rf(train_x, train_y, dev_x, n_feat=80):
    clf = RandomForestClassifier(n_estimators=300).fit(train_x, train_y)
    feat_imp = clf.feature_importances_
    imp_idx = np.argsort(-np.abs(feat_imp))
    train_x = train_x[:, imp_idx[:n_feat], ]
    dev_x = dev_x[:, imp_idx[:n_feat]]
    return RandomForestClassifier(n_estimators=300).fit(train_x, train_y), train_x, dev_x


# def permutation_importance(rf, x_train, y_train, x_test, y_test):
#     # calculate permutation importance for test data
#     result_test = permutation_importance(
#         rf, x_test, y_test, n_repeats=20, random_state=42, n_jobs=2
#     )
#
#     sorted_importances_idx_test = result_test.importances_mean.argsort()
#     importances_test = pd.DataFrame(
#         result_test.importances[sorted_importances_idx_test].T,
#         columns=X.columns[sorted_importances_idx_test],
#     )
#
#     # calculate permutation importance for training data
#     result_train = permutation_importance(
#         rf, x_train, y_train, n_repeats=20, random_state=42, n_jobs=2
#     )
#
#     sorted_importances_idx_train = result_train.importances_mean.argsort()
#     importances_train = pd.DataFrame(
#         result_train.importances[sorted_importances_idx_train].T,
#         columns=X.columns[sorted_importances_idx_train],
#     )
#
#     f, axs = plt.subplots(1, 2, figsize=(15, 5))
#
#     importances_test.plot.box(vert=False, whis=10, ax=axs[0])
#     axs[0].set_title("Permutation Importances (test set)")
#     axs[0].axvline(x=0, color="k", linestyle="--")
#     axs[0].set_xlabel("Decrease in accuracy score")
#     axs[0].figure.tight_layout()
#
#     importances_train.plot.box(vert=False, whis=10, ax=axs[1])
#     axs[1].set_title("Permutation Importances (train set)")
#     axs[1].axvline(x=0, color="k", linestyle="--")
#     axs[1].set_xlabel("Decrease in accuracy score")
#     axs[1].figure.tight_layout()


def one_pass_exp(args, X, Y, F1, vocab, train_test_split):
    train_x, train_y, dev_x, dev_y, train_F1, dev_F1 = apply_train_test_split(X, Y, F1, train_test_split,
                                                                              force_dev_size=args.force_dev_size)
    # print(train_x.shape)
    # print(train_x)
    # print(train_x[0])

    majority_acc = max(np.sum(dev_y == 0), np.sum(dev_y == 1)) / dev_y.size

    if args.do_maxprob:
        train_acc, best_threshold = train_max_accuracy(train_x, train_y)
        # print("dev x", dev_x)
        # print("dev y", dev_y)
        dev_acc, preds = test_max_accuracy(dev_x, dev_y, best_threshold)
        dev_auc = f1auc_score(dev_x, dev_F1)
        _macro_ce = macro_ce(preds, dev_y, dev_x)
        # print("macro ce", _macro_ce)
        return majority_acc, train_acc, dev_acc, dev_auc, f1_prob_curve(dev_F1, dev_x.flatten()), None, _macro_ce

    if args.model == 'lr':
        clf = LogisticRegression(C=1000.0, max_iter=1000, solver='saga').fit(train_x, train_y)
    elif args.model == 'rf':
        # print("in rf")
        # clf = KNeighborsClassifier(n_neighbors=10).fit(train_x, train_y)
        # clf = RandomForestClassifier(n_estimators=200, max_depth=5).fit(train_x, train_y)
        feature_names = [vocab.get_word(i) for i in range(len(dev_x[0]))]
        # print(feature_names)

        clf = RandomForestClassifier(
            n_estimators=args.arg_n_tree,
            # min_samples_split=3,
            max_depth=args.arg_max_depth).fit(train_x, train_y)



        # clf = RandomForest    Regressor(n_estimators=100, max_depth=3).fit(train_x, train_F1)
        # clf, train_x, dev_x = selection_based_rf(train_x, train_y, dev_x)
        # rfe = RFECV(clf, scoring="neg_mean_squared_error")
        # rfe.fit(train_x, train_y)
        # selected_features = [rfe.get_support()]
        # print("selected_features:", selected_features)
        # Get feature importance scores
        # importances = clf.feature_importances_
        #
        # # Sort feature importances in descending order
        # indices = np.argsort(importances)[::-1]
        #
        # # Print the feature ranking
        # print("Feature ranking:")
        # for f in range(X.shape[1]):
        #     print("%d. %s (%f)" % (f + 1, indices[f], importances[indices[f]]))
        # print(train_x.columns)

    elif args.model == 'svm':
        clf = SVC(C=10.0).fit(train_x, train_y)
    elif args.model == 'gdbt':
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=10).fit(train_x, train_y)
    else:
        raise RuntimeError('Model not supported')
    # clf = DecisionTreeClassifier(max_depth=3).fit(train_x, train_y)
    # clf = GradientBoostingClassifier(random_state=args.seed, n_estimators=1
    train_pred = clf.predict(train_x)
    dev_pred = clf.predict(dev_x)
    # print('Label Distribution', np.sum(dev_y == 0)/dev_y.size, np.sum(dev_y == 1)/dev_y.size)
    # print('Train ACC', np.sum(train_pred == train_y)/train_pred.size,
    #     'Dev Acc', np.sum(dev_pred == dev_y) / dev_pred.size)
    # interp_calibrator_model(clf, vocab)
    train_acc = np.sum(train_pred == train_y) / train_pred.size
    dev_acc = np.sum(dev_pred == dev_y) / dev_pred.size
    dev_score = clf.predict_proba(dev_x)[:, 1]
    # dev_auc = auc_score(dev_score, dev_y)
    # print(dev_score)
    # print(dev_F1)
    dev_auc = f1auc_score(dev_score, dev_F1)
    f1_curve = f1_prob_curve(dev_F1, clf.predict_proba(dev_x)[:, 1])
    # f1_curve = f1_prob_curve(dev_F1, clf.predict(dev_x))
    # imp = get_feature_importances(clf, dev_x, dev_y)
    # feats = vocab.get_names()
    # for k,v in zip(feats, imp):
    #     print(k, v)
    # print(clf.predict_proba(dev_x)[:, 1])
    # num_correct = np.sum(dev_pred == dev_y)
    _macro_ce = macro_ce(dev_pred, dev_y, clf.predict_proba(dev_x)[:, 1])
    # print("Dev x: ", dev_x)
    if args.show_imp:
        outputs = []
        selected_indices = [i for i in range(len(dev_pred)) if dev_pred[i] == dev_y[i]]
        # Create Tree Explainer object that can calculate shap values
        explainer = shap.TreeExplainer(clf)
        shap_values_all = explainer.shap_values(dev_x)[1]
        shap.summary_plot(shap_values_all[selected_indices, :], dev_x[selected_indices, :], feature_names=feature_names)
        # print("incorrect_indices", incorrect_indices)
        for i, idx  in tqdm(enumerate(selected_indices)):
            # Calculate Shap values
            choosen_instance = dev_x[idx]
            shap_values = explainer.shap_values(choosen_instance)[1]
            # print(shap_values)
            # shap.initjs()
            # shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)
            # break

            # print(dev_x[incorrect_indices[0]])
            # X_incorrect = X[incorrect_indices, :]
            # feature_importances = clf.feature_importances_
            # print(feature_importances)
            # importances_incorrect = feature_importances[incorrect_indices]
            # print(importances_incorrect)

            # train features
            # imp = permutation_importance(clf, dev_x, dev_y, n_repeats=10, random_state=42, n_jobs=2)

            # Identify the misclassified samples
            # misclassified = dev_x[i]
            # print("Mis", misclassified)
            # print(misclassified[0])
            # print(dev_x[idx])
            # print(dev_y[idx])

            # Estimate the feature importance for the misclassified samples using permutation importance
            # imp = permutation_importance(
            #     clf, dev_x[idx].reshape(1, -1), dev_y[idx].reshape(1, -1), n_repeats=10, random_state=42, n_jobs=2
            # )
            #
            # # # Extract the feature importance scores
            # misclassified_importance = imp.importances_mean
            # # print(misclassified_importance)
            # # imp_idx = np.argsort(-np.abs(misclassified_importance))
            # # Sort the feature importances in descending order
            sorted_indices = np.argsort(shap_values)[::-1]
            # # print("Sort ind: ", sorted_indices)
            importance = []
            for rank, j in enumerate(sorted_indices[:20]):
                # print('Feat Rank:', rank, vocab.get_word(i), agg_feat_imp[i])
                # print(rank, vocab.get_word(j), misclassified_importance[j])
                importance.append((rank, vocab.get_word(j), shap_values[j]))
            # print(importance)
            # break
            #
            output = collections.OrderedDict()
            output["idx"] = idx
            output["importance"] = importance
            outputs.append(output)
            # print(output)
        # BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"
        # save_file_name = "lime_squad_adv_classified_using_shap"
        # output_nbest_file = os.path.join(BASE_PATH, f"predictions_{save_file_name}.json")
        # with open(output_nbest_file, "w") as writer:
        #     writer.write(json.dumps(outputs, indent=4) + "\n")

    return majority_acc, train_acc, dev_acc, dev_auc, f1_curve, get_feature_importances(clf), _macro_ce


def macro_ce(preds, golds, probs):
    """Macro CE"""
    ice_pos, ice_neg = 0, 0
    num_pos, num_neg = 0, 0
    for i in range(len(preds)):
        pred = preds[i]
        pred_proba = probs[i]
        gold = golds[i]
        if pred==gold:
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


def gen_predefined_train_test_splits(baseids, n_run, train_size):
    num_data = len(baseids)

    baseid_indexer = OrderedDict()
    for i, b in enumerate(baseids):
        if b in baseid_indexer:
            baseid_indexer[b].append(i)
        else:
            baseid_indexer[b] = [i]
    num_base = len(baseid_indexer)
    print('Number of Exs', num_data, 'Number of Unique Base', num_base)

    predefined_permutations = [np.random.permutation(num_base) for _ in range(n_run)]
    baseid_groups = list(baseid_indexer.values())
    num_train = train_size

    # num_data_cf = len(cf_baseids)
    # baseid_indexer_cf = OrderedDict()
    # for i, b in enumerate(cf_baseids):
    #     if b in baseid_indexer_cf:
    #         baseid_indexer_cf[b].append(i)
    #     else:
    #         baseid_indexer_cf[b] = [i]
    # num_base_cf = len(baseid_indexer_cf)
    # print('Number of cf Exs', num_data_cf, 'Number of Unique cf Base', num_base_cf)

    # predefined_permutations_cf = [np.random.permutation(num_base_cf) for _ in range(n_run)]
    # baseid_groups_cf = list(baseid_indexer_cf.values())
    # num_train_cf = train_size

    splits = []
    for perm in predefined_permutations:
        flat_idx = list(chain(*[baseid_groups[i] for i in perm]))
        # flat_idx_cf = list(chain(*[baseid_groups_cf[i] for i in perm_cf]))
        # print(flat_idx_cf[:num_train_cf])
        train_index = flat_idx[:num_train] #+ flat_idx_cf[:num_train_cf]
        # train_index.append(flat_idx_cf[:num_train_cf])
        dev_index = flat_idx[num_train:]
        splits.append((train_index, dev_index))
    print('Train Size', len(splits[0][0]), 'Dev Size', len(splits[0][1]))
    return splits


def prediction_direction_of_feat(X, Y, F1):
    clf = LogisticRegression(C=1000.0, max_iter=10000).fit(X, Y)
    feat_imp = clf.coef_.flatten()
    train_pred = clf.predict(X)
    train_acc = np.sum(train_pred == Y) / train_pred.size
    print('Linear ACC:', train_acc)
    return feat_imp


def quantify_colum(x, k=2, method='val'):
    if method == 'val':
        interval = np.arange(k) / k
        vals = (np.max(x) - np.min(x)) * interval + np.min(x)
    if method == 'percent':
        q = 100 * np.arange(k) / k
        vals = np.percentile(x, q)

    new_x = np.zeros_like(x)

    for i, v in enumerate(vals):
        new_x[x >= v] = i / k
    return new_x


def quantify_dataset(X, vocab, k, type):
    # print(X.shape)
    for i in range(X.shape[1]):
        fname = vocab.get_word(i)
        if 'BASELINE' in fname:
            continue
        if 'BOW' in fname:
            continue
        # print(fname)
        X[:, i] = quantify_colum(X[:, i], k=k, method=type)
    return X, vocab


def main():
    args = _parse_args()
    # print(args)
    np.random.seed(args.seed)
    # data = load_bin('./calib_files/squad_addsent-dev_shap_calib_data.bin')
    for model in ["llama", "gpt_neox", "flan_ul2"]:
        data = load_bin(f'./src/data/hotpot_qa/calib_data_simple_grads_{model}_mod.bin')
        def take(n, iterable):
            """Return the first n items of the iterable as a list."""
            return list(islice(iterable, n))

        # data = take(1000, data.items())
        data = proc_input_data(args, dict(data))
        print("Total samples: ", len(data))

        X, Y, F1, vocab = make_np_dataset(data)
        # print(Y)
        # print(np.sum(Y))
        if args.quant_size > 0:
            X, vocab = quantify_dataset(X, vocab, args.quant_size, args.quant_type)
        # print(X.shape, Y.shape)
        # print(np.sum(Y == 0)/ Y.size, np.sum(Y == 1)/ Y.size)

        if args.dataset == 'squad':
            # filter cf baseids
            cf_baseids = [sample for sample in data if "-cf-" in sample]
            cf_baseids_filtered = [sample.split('-')[0] for sample in cf_baseids if "turk" in sample]
            print(len(cf_baseids_filtered))
            data_without_cf = [sample for sample in data if sample not in cf_baseids]
            print("Number of counterfactuals: ", len(cf_baseids))
            baseids = [k.split('-')[0] for k in data_without_cf if args.dataset == 'squad' and "turk" in k]

        elif args.dataset == 'trivia':
            # filter cf baseids
            # cf_baseids = [sample for sample in data if "-cf-" in sample]
            # print("Number of counterfactuals: ", len(cf_baseids))
            cf_baseids_filtered = [sample.split('-')[0] for sample in data if '-cf-' in sample]
            print("Number of filtered counterfactuals: ", len(cf_baseids_filtered))
            data_without_cf = [sample for sample in data if '-cf-' not in sample]
            # print(data_without_cf)
            print("Number of original samples: ", len(data_without_cf))
            baseids = [k for k in data_without_cf]

        # print(len(baseids))
        # print(baseids)
        predefined_splits = gen_predefined_train_test_splits(baseids, args.n_run, args.train_size)
        agg_results = []
        for train_test_split in tqdm(predefined_splits, total=len(predefined_splits), desc='Running Exp...'):
            if args.include_cf:
                indices =[]
                train_set = train_test_split[0]
                test_set = train_test_split[1]
                for i, b in enumerate(train_set):
                    if baseids[b] in cf_baseids_filtered:
                        indices.append(baseids[b])

                cf_indices = []
                for i, b in enumerate(data):
                    # print(b)
                    if "-cf-" in b and b.split("-")[0] in indices:
                        cf_indices.append(i)
                # print(cf_indices)
                train_set.extend(cf_indices)

            # print(train_set)
            # print(cf_indices)
            # print(len(cf_indices))
            # c=0
            # for i in train_set:
            #     if i in cf_indices:
            #         c+=1
            # print(c)
            # shuffle cf_indices
                np.random.shuffle(train_set)
                augmented_train_set = (train_set, test_set)
                print(len(train_set), len(test_set))
            else:
                augmented_train_set = train_test_split
            # print(len(augmented_train_set[0]), len(augmented_train_set[1]))
            agg_results.append(one_pass_exp(args, X, Y, F1, vocab, augmented_train_set))

        # print(agg_results)
        agg_base_acc = np.array([x[0] for x in agg_results])
        agg_train_acc = np.array([x[1] for x in agg_results])
        agg_dev_acc = np.array([x[2] for x in agg_results])
        agg_auc = np.array([x[3] for x in agg_results])
        agg_f1_curve = np.array([x[4] for x in agg_results]).mean(axis=0)
        agg_macro_ce = np.array([x[6] for x in agg_results]).mean()
        print('AVG MACRO CE {:.3f}'.format(agg_macro_ce))

        print('AVG MAJORITY ACC {:.3f}'.format(agg_base_acc.mean()))
        print('AVG TRAIN ACC {:.3f}'.format(agg_train_acc.mean()))
        print('AVG DEV ACC: {:.3f} +/- {:.3f}, AUC: {:.3f}, MAX: {:.3f}, MIN: {:.3f}'.format(agg_dev_acc.mean(),
                                                                                             agg_dev_acc.std(),
                                                                                             agg_auc.mean(),
                                                                                             agg_dev_acc.max(),
                                                                                             agg_dev_acc.min()))

        # print numbers for copy paste
        exp_numbers = [agg_base_acc.mean(), agg_dev_acc.mean(), agg_auc.mean()] + agg_f1_curve.tolist()
        print(','.join(['{:.3f}'.format(x) for x in exp_numbers]))
        exp_numbers = exp_numbers[1:]
        print(','.join(['{:.1f}'.format(a * 100) for a in exp_numbers]))
        if agg_results[0][5] is not None and args.show_imp:
            agg_feat_imp = np.array([x[5] for x in agg_results])
            # print(agg_feat_imp.shape)
            agg_feat_imp = np.mean(agg_feat_imp, axis=0)
            imp_idx = np.argsort(-np.abs(agg_feat_imp))
            # pred_direction = prediction_direction_of_feat(X, Y, F1)
            for rank, i in enumerate(imp_idx[:100]):
                # print(imp_idx[i])
                # print('Feat Rank:', rank, vocab.get_word(i), agg_feat_imp[i])
                print(rank, vocab.get_word(i), agg_feat_imp[i])


if __name__ == "__main__":
    main()
