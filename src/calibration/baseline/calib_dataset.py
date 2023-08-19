import ast
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import math
import re
import spacy
from spacy.tokens import Doc
import torch
# from supar import Parser

from src.calibration.baseline import common, evaluate, utils

import warnings
# ignore warning
warnings.filterwarnings("ignore")


class CreateFeatures:
    def __init__(self, path, model, dataset, method):
        self.path = path
        self.cls_tokens = ["[CLS]", "<s>"]
        self.sep_tokens = ["[SEP]", "</s>"]
        self.count = 0
        self.nlp = spacy.load("en_core_web_sm")
        self.model = model
        self.dataset = dataset.split("_")[0]
        self.method = method

    def _load_data(self):
        self.tagger_info = utils.load_bin(f"{self.path}/pos_info.bin")
        # self.ent_info = utils.load_bin(f"{self.path}/ent_info.bin")
        self.states = utils.load_bin(f"{self.path}/dense_repr_pca_10_info_{self.model}.bin")
        self.attributions = utils.load_bin(f"{self.path}/{self.method}_info_{self.model}.bin")
        pred_df = pd.read_csv(f"{self.path}/outputs_{self.model}.csv")
        self.preds_info = pred_df.set_index('id').T.to_dict('dict')
        # self.preds_info = utils.read_json('outputs/addsent-dev_squad_predictions.json')

    def featurize(self):
        self._load_data()
        processed_instances = OrderedDict()
        c = 0
        for q_id in tqdm(self.tagger_info, total=len(self.tagger_info), desc='transforming...'):
            tags = self.tagger_info[q_id]
            # entities = self.ent_info[q_id]
            attributions = self.attributions[q_id]
            states = self.states[q_id]
            preds = self.preds_info[q_id]
            processed_instances[q_id] = self.extract_feature_for_instance(attributions, tags, preds, states)
            c+=1
        utils.dump_to_bin(processed_instances, f"{self.path}/calib_data_{self.method}_{self.model}.bin")

    def lemmatize_pos_tag(self, token_tags):
        """
        limit the amount of POS tags to reduce the number of features
        :param token_tags: tagged tokens
        :return:
        """
        tok, pos, tag = token_tags
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
        if pos == 'PUNCT':
            tag = 'PUNCT'
        return tok, pos, tag

    def extract_baseline_feature(self, preds):
        # print(preds)
        predictions = ast.literal_eval(preds["pred_text"])[0]
        # print(predictions)
        feat = common.IndexedFeature()
        # base features
        for rank, p in enumerate(predictions[:5]):
            feat.add(f'BASELINE_PROB_{rank}', p['score'])
        feat.add('BASELINE_CONTEXT_LENGTH', len(preds['context']))
        feat.add('BASELINE_PRED_ANS_LENGTH', len(predictions[0]['answer']))

        top_pred = predictions[0]['answer']
        first_distinct_prob = 0
        for i, p in enumerate(predictions[1:]):
            overlapping = evaluate.f1_score(p['answer'], top_pred)
            if overlapping > 0:
                continue
            first_distinct_prob = p['score']
            # print(i+1, top_pred, p['answer'], first_distinct_prob)
        feat.add('FIRST_DISTINCT_PROB', first_distinct_prob)
        return feat

    def source_of_token(self, idx, tok, context_start, ans_range):
        if tok in self.cls_tokens or tok in self.sep_tokens:
            return 'S'
        if idx >= 1 and idx < context_start:
            return 'Q'
        if idx >= ans_range[0] and idx <= ans_range[1]:
            # print("Answer token:", tok)
            return 'A'
        return 'C'

    def nan_check(self, result):
        for i, (key, value) in enumerate(result):
            if math.isnan(float(value)):
                result[i][1] = 0
        return result

    def count_syllables(self, word: str):
        return len(
            re.findall('(?!e$)[aeiouy]+', word, re.I) +
            re.findall('^[^aeiouy]*e$', word, re.I)
        )

    def handle_zero_division(self, n, d):
        return n / d if d else 0

    def extract_state_features(self, attr, tokens, ans_range, states):
        """
        Extract Q, C, and A features from model hidden states
        """
        feat = common.IndexedFeature()
        context_start = tokens.index(self.sep_tokens[1])
        start, end = ans_range
        ans_indices = list(range(start, end + 1))
        q_states = states[:, 1:context_start, :]
        c_states = states[:, context_start + 2: -1, :]
        torch.set_printoptions(threshold=10_000)
        # remove answer states from context states
        c_states_before = c_states[:, :start - (context_start + 2), :]
        c_states_after = c_states[:, end + 1 - (context_start + 2):, :]
        final_c_states = torch.cat((c_states_before, c_states_after), dim=1)
        a_states = states[:, start:end + 1, :]
        attribution = np.array(attr["attributions"])
        question_attr = attribution[1:context_start]
        context_attr = attribution[context_start + 2: -1]
        c_attr_before = context_attr[:start - (context_start + 2)]
        c_attr_after = context_attr[end + 1 - (context_start + 2):]
        final_c_attr = np.concatenate((c_attr_before, c_attr_after), axis=0)
        ans_attr = attribution[start: end + 1]

        ques_sorted_indices = np.argsort(question_attr)
        # take topk indices
        topk_ques_indices = ques_sorted_indices[-int(ques_sorted_indices.shape[0] * 0.10):]
        ques_topk_states = np.take(q_states, topk_ques_indices, axis=1)

        topc_percent = 0.10
        cxt_sorted_indices = np.argsort(final_c_attr)
        cxt_top_k_indices = cxt_sorted_indices[-int(cxt_sorted_indices.shape[0] * topc_percent):]
        cxt_topk_states = np.take(final_c_states, cxt_top_k_indices, axis=1)

        topa_percent = 0.20
        ans_sorted_indices = np.argsort(ans_attr)
        ans_top_k_indices = ans_sorted_indices[-int(ans_sorted_indices.shape[0] * topa_percent):]
        ans_topk_states = np.take(a_states, ans_top_k_indices, axis=1)

        q_rep = torch.mean(ques_topk_states, dim=1).tolist()[0]
        # print(q_rep.shape)
        c_rep = torch.mean(cxt_topk_states, dim=1).tolist()[0]
        # print(c_rep.shape)
        a_rep = torch.mean(ans_topk_states, dim=1).tolist()[0]

        # for i, value in enumerate(q_rep):
        #     if math.isnan(value):
        #         continue
        #     feat.add_new(f"REPR_TOPK_Q_{i}", value)
        for i, value in enumerate(c_rep):
            if math.isnan(value):
                continue
            feat.add_new(f"REPR_TOPK_C_{i}", value)
            # print(value)
        for i, value in enumerate(a_rep):
            if math.isnan(value):
                continue
            feat.add_new(f"REPR_TOPK_A_{i}", value)

        return feat

    def extract_bow_feature(self, words, tags, ans_range):
        feat = common.IndexedFeature()
        context_start = words.index(self.sep_tokens[1])
        for i, (i_token, i_pos, i_tag) in enumerate(tags):
            i_src = self.source_of_token(i, i_token, context_start, ans_range)
            if i_src == 'Q' or i_src == 'A' or i_src == 'C':
                # print('BOW_{}_{}'.format(i_src, i_tag))
                feat.add('BOW_{}_{}'.format(i_src, i_tag))
                feat.add('BOW_IN_{}'.format(i_tag))
        return feat

    def merge_attribution_by_segments(self, attribution, segments):
        new_val = []
        for a, b in segments:
            new_val.append(np.sum(attribution[a:b], axis=0))
        importance = np.stack(new_val, axis=0)
        return importance

    def aggregate_token_attribution(self, attr, tags, polarity):
        # attr = ast.literal_eval(attr)
        # print(attr)
        attribution_val = np.array(attr["attributions"])
        if polarity == 'POS':
            attribution_val[attribution_val < 0] = 0
        elif polarity == 'NEG':
            attribution_val[attribution_val > 0] = 0
        elif polarity == 'NEU':
            pass
        else:
            raise RuntimeError('Invalid polarity')

        attribution_val = self.merge_attribution_by_segments(attribution_val, tags['segments'])
        assert attribution_val.shape[0] == len(tags['segments'])
        # normalize
        attribution_val = attribution_val / np.sum(attribution_val)
        # print(attribution_val)
        return attribution_val

    def normalize_token_attr(self, feat, attributions, norm_method=None):
        if norm_method is None:
            return feat
        if norm_method == 'all':
            sum_v = np.sum(attributions)
            for k in feat.data:
                feat.data[k] = feat.data[k] / sum_v if sum_v != 0 else 0
            return feat
        if norm_method == 'counted':
            sum_v = sum(feat.data.values())
            for k in feat.data:
                feat.data[k] = feat.data[k] / sum_v if sum_v != 0 else 0
            return feat
        raise RuntimeError(norm_method)

    def extract_token_attr_feature_in_question(self, words, tags, attributions, include_punct=False):
        context_start = words.index(self.sep_tokens[1])
        tags = tags[1:context_start]
        attributions = attributions[1:context_start]

        feat = common.IndexedFeature()
        unnorm = common.IndexedFeature()
        sum_v = 0
        for i, (token, pos, tag) in enumerate(tags):
            if tag:
                v = attributions[i]
                if pos == 'PUNCT' and not include_punct:
                    continue
                feat.add('NORMED_TOK_Q_' + tag, v)
                unnorm.add('UNNORM_TOK_Q_' + tag, v)
                sum_v += v

        feat = self.normalize_token_attr(feat, attributions)
        feat.add_set(unnorm)
        feat.add('SUM_TOK_Q', sum_v)
        return feat

    def extract_token_attr_feature_in_context(self, words, tags, attributions, include_punct=False):
        context_start = words.index(self.sep_tokens[1])
        tags = tags[context_start + 2: -1]
        attributions = attributions[context_start + 2: -1]

        feat = common.IndexedFeature()
        unnorm = common.IndexedFeature()
        sum_v = 0
        for i, (token, pos, tag) in enumerate(tags):
            if tag:
                v = attributions[i]
                if pos == 'PUNCT' and not include_punct:
                    continue
                feat.add('NORMED_TOK_C_' + tag, v)
                feat.add('UNNORM_TOK_C_' + tag, v)
                sum_v += v

        feat = self.normalize_token_attr(feat, attributions)
        feat.add_set(unnorm)
        feat.add('SUM_TOK_C', sum_v)
        return feat

    def extract_token_attr_feature_in_input(self, words, tags, attributions, ans_range, include_punct=False):
        feat = common.IndexedFeature()
        context_start = words.index(self.sep_tokens[1])
        normed_a_feat = common.IndexedFeature()
        unnormed_a_feat = common.IndexedFeature()
        for i, (token, pos, tag) in enumerate(tags):
            if tag:
                v = attributions[i]
                if pos == 'PUNCT' and not include_punct:
                    continue
                feat.add('TOK_IN_' + tag, v)

                if i >= ans_range[0] and i <= ans_range[1]:
                    unnormed_a_feat.add('UNNORM_TOK_A_' + tag, v)
                    normed_a_feat.add('NORMED_TOK_A_' + tag, v)

        feat.add_set(self.normalize_token_attr(normed_a_feat, [], 'counted'))
        feat.add_set(unnormed_a_feat)
        return feat

    def extract_token_attr_stats_in_input(self, words, tags, attributions, part):
        feat = common.IndexedFeature()
        context_start = words.index(self.sep_tokens[1])
        if part == 'Q':
            tags = tags[1:context_start]
            attributions = attributions[1:context_start]
        if part == 'C':
            tags = tags[context_start + 2: -1]
            attributions = attributions[context_start + 2: -1]

        feat.add('STAT_MEAN_' + part, attributions.mean())
        feat.add('STAT_STD_' + part, attributions.std())
        return feat

    def extract_polarity_feature(self, attr, tags, words, tags_for_tok, ans_index, polarity, include_basic=True,
                                 include_stats=False):
        named_feat = common.IndexedFeature()
        token_attribution = self.aggregate_token_attribution(attr, tags, polarity)

        if include_basic:
            named_feat.add_set(self.extract_token_attr_feature_in_question(
                words,
                tags_for_tok,
                token_attribution))
            named_feat.add_set(self.extract_token_attr_feature_in_context(
                words,
                tags_for_tok,
                token_attribution))
            named_feat.add_set(
                self.extract_token_attr_feature_in_input(
                    words,
                    tags_for_tok,
                    token_attribution,
                    ans_index))

        if include_stats:
            named_feat.add_set(self.extract_token_attr_stats_in_input(
                words,
                tags_for_tok,
                token_attribution,
                'Q'))
            named_feat.add_set(self.extract_token_attr_stats_in_input(
                words,
                tags_for_tok,
                token_attribution,
                'C'))
            named_feat.add_set(self.extract_token_attr_stats_in_input(
                words,
                tags_for_tok,
                token_attribution,
                'IN'))
        named_feat.add_prefix(polarity + '_')
        return named_feat

    def extract_feature_for_instance(self, attr, tags, preds, states):
        predictions = ast.literal_eval(preds["pred_text"])[0]
        pred_text = predictions[0]['answer']
        if self.dataset == "squad":
            gold_text = ast.literal_eval(preds['gold_text'])
        else:
            gold_text = preds['gold_text']

        exact_match = evaluate.get_score(
            metric="exact_match",
            pred_text=pred_text,
            gold_text=gold_text
        )
        f1 = evaluate.get_score(metric="f1", pred_text=pred_text, gold_text=gold_text)
        calib_label = 1 if exact_match > 0 else 0
        # print(calib_label)

        # baseline features
        named_feat = common.IndexedFeature()
        named_feat.add_set(self.extract_baseline_feature(preds))

        start_index, end_index = ast.literal_eval(preds["answer_start"])[0],\
                                 ast.literal_eval(preds["answer_end"])[0]
        segments = tags["segments"]
        transformed_start_idx = 0
        while segments[transformed_start_idx + 1][0] <= start_index:
            transformed_start_idx += 1
        transformed_end_idx = 0
        while segments[transformed_end_idx][1] < (end_index + 1):
            transformed_end_idx += 1
        ans_range = (transformed_start_idx, transformed_end_idx)
        # syntactic features
        words, tags_for_tok = tags['words'], tags['tags']
        # words = [w for w in words if w != " "]
        tags_for_tok = [self.lemmatize_pos_tag(x) for x in tags_for_tok]
        # ents_for_tok = entities['ents']

        named_feat.add_set(self.extract_state_features(attr, words, ans_range, states))
        named_feat.add_set(self.extract_bow_feature(words, tags_for_tok, ans_range))
        named_feat.add_set(self.extract_polarity_feature(attr, tags, words, tags_for_tok, ans_range, 'NEU'))
        # print({'feature': named_feat, 'label': calib_label, 'f1_score': f1})
        return {'feature': named_feat, 'label': calib_label, 'f1_score': f1}


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Passing arguments for model, tokenizer, and dataset.")

    parser.add_argument("--model_name", type=str, required=False, help="Specify the model to use.")
    parser.add_argument("--dataset", type=str, required=True, help="Specify the dataset to use.")
    parser.add_argument("--method", type=str, required=True, help="Specify the explanation method to use.")
    args = parser.parse_args()

    data_path = f"./src/data/{args.dataset}"
    feat = CreateFeatures(path=data_path, model=args.model_name, dataset=args.dataset, method=args.method)
    feat.featurize()
