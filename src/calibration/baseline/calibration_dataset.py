import os
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import math
import re
import spacy
from spacy.tokens import Doc
# from supar import Parser
import traceback

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
        # self.supar = Parser.load('crf-con-en')

    def _load_data(self):
        if self.dataset == "squad":
            pos_file = "pos_info.bin"
        else:
            pos_file = "pos_info_wo_space.bin"
        if self.model == "base":
            self.tagger_info = utils.load_bin(f"{self.path}/{pos_file}")
            self.attributions = self.build_file_dict(dataset=self.dataset, split="addsent-dev", method=self.method)
            # self.attributions = utils.load_bin(f"{self.path}/attn_info_with_easy_0.75.bin")
            self.states = utils.load_bin(f"{self.path}/dense_repr_pca_10_info_base.bin")
            self.preds_info = utils.read_json(
                f"{self.path}/nbest_predictions_roberta_squad_top20.json")

        elif self.model == "rag":
            self.tagger_info = utils.load_bin(f"{self.path}/{pos_file}")
            self.attributions = self.build_file_dict(dataset=self.dataset, split="addsent-dev", method=self.method)
            # self.attributions = utils.load_bin(f"{self.path}/attn_info_with_easy_0.75.bin")
            self.states = utils.load_bin(f"{self.path}/dense_repr_pca_10_info_rag.bin")
            self.preds_info = utils.read_json(
                f"{self.path}/nbest_predictions_roberta-squad-t5-squad-cfs-seed-42.json")

        elif self.model == "llama":
            self.tagger_info = utils.load_bin(f"{self.path}/{pos_file}")
            self.attributions = self.build_file_dict(dataset=self.dataset, split="addsent-dev", method=self.method)
            # self.attributions = utils.load_bin(f"{self.path}/attn_info_with_easy_0.75.bin")
            self.states = utils.load_bin(f"{self.path}/dense_repr_pca_10_info_llama.bin")
            self.preds_info = utils.read_json(
                f"{self.path}/nbest_predictions_roberta-squad-llama-context-rel-seed-42.json")

        elif self.model == "gpt_neox":
            self.tagger_info = utils.load_bin(f"{self.path}/{pos_file}")
            self.attributions = self.build_file_dict(dataset=self.dataset, split="addsent-dev", method=self.method)
            # self.attributions = utils.load_bin(f"{self.path}/attn_info_with_easy_0.75.bin")
            self.states = utils.load_bin(f"{self.path}/dense_repr_pca_10_info_gpt_neox.bin")
            self.preds_info = utils.read_json(
                f"{self.path}/nbest_predictions_roberta-squad-gpt-neox-context-rel-seed-42.json")

        elif self.model == "flan_ul2":
            self.tagger_info = utils.load_bin(f"{self.path}/{pos_file}")
            self.attributions = self.build_file_dict(dataset=self.dataset, split="addsent-dev", method=self.method)
            # self.attributions = utils.load_bin(f"{self.path}/attn_info_with_easy_0.75.bin")
            self.states = utils.load_bin(f"{self.path}/dense_repr_pca_10_info_flan_ul2.bin")
            self.preds_info = utils.read_json(f"{self.path}/nbest_predictions_roberta-squad-flan-ul2-context-rel-noise-seed-42.json")

    def load_interp_info(self, file_dict, qas_id):
        return torch.load(file_dict[qas_id])

    def build_file_dict(self, dataset, split, method):
        # hard-coded path here: be careful
        # prefix = 'squad_sample-addsent_roberta-base'
        prefix = f'{dataset}/dev/roberta'
        if self.model == "gpt_neox" or self.model == "llama":
            exp_file_name = f"exp_roberta_{self.model}_context_rel"
        elif self.model == "flan_ul2":
            exp_file_name = f"exp_roberta_{self.model}_context_noise_rel"
        elif self.model == "rag":
            exp_file_name = f"exp_roberta_{self.model}"
        elif self.model == "base":
            exp_file_name = f"exp_roberta_{self.model}"
        fnames = os.listdir(os.path.join(exp_file_name, method, prefix))
        # print(fnames)
        qa_ids = [x.split('.')[0] for x in fnames]
        # exp_roberta_flan_ul2_context_noise_rel
        fullnames = [os.path.join(exp_file_name, method, prefix, x) for x in fnames]
        return dict(zip(qa_ids, fullnames))

    def featurize(self):
        self._load_data()
        processed_instances = OrderedDict()
        c=0
        total_count = 0
        # print(self.attributions)
        for q_id in tqdm(self.tagger_info, total=len(self.tagger_info), desc='transforming...'):
            total_count+=1
            try:
                tags = self.tagger_info[q_id]
                # print(tags)
                interp = self.load_interp_info(self.attributions, q_id)
                # print(interp)
                states = self.states[q_id]
                preds = self.preds_info[q_id]
                processed_instances[q_id] = self.extract_feature_for_instance(interp, tags, preds, states)
            except Exception as e:
                c+=1
                print(f"An exception occurred: {e}")
                print("Total", total_count)
                print(q_id)
                traceback.print_exc()
                continue
            # if total_count==1:
            #     break
        print("Total instances not processed: ", c)
        utils.dump_to_bin(processed_instances, f"{self.path}/calib_data_{method}_{model}_mod.bin")

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
        if pos == 'PUNCT':  #or tag in [":"]:
            tag = 'PUNCT'
        # if tag == '$':
        #     tag = 'SYM'
        # if tag == "``" or tag == "''":
        #     tag = "AUX"
        return tok, pos, tag

    def extract_baseline_feature(self, preds, attr):
        feat = common.IndexedFeature()
        # base features
        for rank, p in enumerate(preds[:5]):
            feat.add(f'BASELINE_PROB_{rank}', p['probability'])
        feat.add('BASELINE_CONTEXT_LENGTH', len(attr['example'][1]))
        feat.add('BASELINE_PRED_ANS_LENGTH', len(preds[0]['text']))

        top_pred = preds[0]['text']
        first_distinct_prob = 0
        for i, p in enumerate(preds[1:]):
            overlapping = evaluate.f1_score(p['text'], top_pred)
            if overlapping > 0:
                continue
            first_distinct_prob = p['probability']
            # print(i+1, top_pred, p['text'], first_distinct_prob)
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


    def pos_tags(self, tokens):
        words = [x.lstrip() for x in tokens]
        spaces = [False if i == len(tokens) - 1
                  else tokens[i + 1][0] == ' ' for i in range(len(tokens))]

        valid_idx = [i for i, w in enumerate(words) if len(w)]
        words = [words[i] for i in valid_idx]
        spaces = [spaces[i] for i in valid_idx]
        doc = Doc(self.nlp.vocab, words=words, spaces=spaces)
        processed_tokens = self.nlp(doc)
        n_sent, n_token = 0, 0
        for sent in processed_tokens.sents:
            n_sent += 1
            for token in sent:
                n_token += 1

        to_NoTag_C = 0
        to_VeTag_C = 0
        to_AjTag_C = 0
        to_AvTag_C = 0
        to_SuTag_C = 0
        to_CoTag_C = 0
        to_ContW_C = 0
        to_FuncW_C = 0

        for token in processed_tokens:
            if token.pos_ == "NOUN" or token.pos_ == "VERB" or token.pos_ == "NUM" or token.pos_ == "ADJ" or token.pos_ == "ADV":
                to_ContW_C += 1
            else:
                to_FuncW_C += 1

            if token.pos_ == "NOUN":
                to_NoTag_C += 1
            if token.pos_ == "VERB":
                to_VeTag_C += 1
            if token.pos_ == "ADJ":
                to_AjTag_C += 1
            if token.pos_ == "ADV":
                to_AvTag_C += 1
            if token.pos_ == "SCONJ":
                to_SuTag_C += 1
            if token.pos_ == "CCONJ":
                to_CoTag_C += 1

        result = {
            "to_NoTag_C": float(to_NoTag_C),
            "as_NoTag_C": float(self.handle_zero_division(to_NoTag_C, n_sent)),
            "at_NoTag_C": float(self.handle_zero_division(to_NoTag_C, n_token)),
            "ra_NoAjT_C": float(self.handle_zero_division(to_NoTag_C, to_AjTag_C)),
            "ra_NoVeT_C": float(self.handle_zero_division(to_NoTag_C, to_VeTag_C)),
            "ra_NoAvT_C": float(self.handle_zero_division(to_NoTag_C, to_AvTag_C)),
            "ra_NoSuT_C": float(self.handle_zero_division(to_NoTag_C, to_SuTag_C)),
            "ra_NoCoT_C": float(self.handle_zero_division(to_NoTag_C, to_CoTag_C)),
            "to_VeTag_C": float(to_VeTag_C),
            "as_VeTag_C": float(self.handle_zero_division(to_VeTag_C, n_sent)),
            "at_VeTag_C": float(self.handle_zero_division(to_VeTag_C, n_token)),
            "ra_VeAjT_C": float(self.handle_zero_division(to_VeTag_C, to_AjTag_C)),
            "ra_VeNoT_C": float(self.handle_zero_division(to_VeTag_C, to_NoTag_C)),
            "ra_VeAvT_C": float(self.handle_zero_division(to_VeTag_C, to_AvTag_C)),
            "ra_VeSuT_C": float(self.handle_zero_division(to_VeTag_C, to_SuTag_C)),
            "ra_VeCoT_C": float(self.handle_zero_division(to_VeTag_C, to_CoTag_C)),
            "to_AjTag_C": float(to_AjTag_C),
            "as_AjTag_C": float(self.handle_zero_division(to_AjTag_C, n_sent)),
            "at_AjTag_C": float(self.handle_zero_division(to_AjTag_C, n_token)),
            "ra_AjNoT_C": float(self.handle_zero_division(to_AjTag_C, to_NoTag_C)),
            "ra_AjVeT_C": float(self.handle_zero_division(to_AjTag_C, to_VeTag_C)),
            "ra_AjAvT_C": float(self.handle_zero_division(to_AjTag_C, to_AvTag_C)),
            "ra_AjSuT_C": float(self.handle_zero_division(to_AjTag_C, to_SuTag_C)),
            "ra_AjCoT_C": float(self.handle_zero_division(to_AjTag_C, to_CoTag_C)),
            "to_AvTag_C": float(to_AvTag_C),
            "as_AvTag_C": float(self.handle_zero_division(to_AvTag_C, n_sent)),
            "at_AvTag_C": float(self.handle_zero_division(to_AvTag_C, n_token)),
            "ra_AvAjT_C": float(self.handle_zero_division(to_AvTag_C, to_AjTag_C)),
            "ra_AvNoT_C": float(self.handle_zero_division(to_AvTag_C, to_NoTag_C)),
            "ra_AvVeT_C": float(self.handle_zero_division(to_AvTag_C, to_VeTag_C)),
            "ra_AvSuT_C": float(self.handle_zero_division(to_AvTag_C, to_SuTag_C)),
            "ra_AvCoT_C": float(self.handle_zero_division(to_AvTag_C, to_CoTag_C)),
            "to_SuTag_C": float(to_SuTag_C),
            "as_SuTag_C": float(self.handle_zero_division(to_SuTag_C, n_sent)),
            "at_SuTag_C": float(self.handle_zero_division(to_SuTag_C, n_token)),
            "ra_SuAjT_C": float(self.handle_zero_division(to_SuTag_C, to_AjTag_C)),
            "ra_SuNoT_C": float(self.handle_zero_division(to_SuTag_C, to_NoTag_C)),
            "ra_SuVeT_C": float(self.handle_zero_division(to_SuTag_C, to_VeTag_C)),
            "ra_SuAvT_C": float(self.handle_zero_division(to_SuTag_C, to_AvTag_C)),
            "ra_SuCoT_C": float(self.handle_zero_division(to_SuTag_C, to_CoTag_C)),
            "to_CoTag_C": float(to_CoTag_C),
            "as_CoTag_C": float(self.handle_zero_division(to_CoTag_C, n_sent)),
            "at_CoTag_C": float(self.handle_zero_division(to_CoTag_C, n_token)),
            "ra_CoAjT_C": float(self.handle_zero_division(to_CoTag_C, to_AjTag_C)),
            "ra_CoNoT_C": float(self.handle_zero_division(to_CoTag_C, to_NoTag_C)),
            "ra_CoVeT_C": float(self.handle_zero_division(to_CoTag_C, to_VeTag_C)),
            "ra_CoAvT_C": float(self.handle_zero_division(to_CoTag_C, to_AvTag_C)),
            "ra_CoSuT_C": float(self.handle_zero_division(to_CoTag_C, to_SuTag_C)),
            "to_ContW_C": float(to_ContW_C),
            "as_ContW_C": float(self.handle_zero_division(to_ContW_C, n_sent)),
            "at_ContW_C": float(self.handle_zero_division(to_ContW_C, n_token)),
            "to_FuncW_C": float(to_FuncW_C),
            "as_FuncW_C": float(self.handle_zero_division(to_FuncW_C, n_sent)),
            "at_FuncW_C": float(self.handle_zero_division(to_FuncW_C, n_token)),
            "ra_CoFuW_C": float(self.handle_zero_division(to_ContW_C, to_FuncW_C)),
        }
        return result

    def extract_pos(self, tokens, ans_range):
        feat = common.IndexedFeature()
        context_start = tokens.index(self.sep_tokens[1])
        question_tokens = tokens[1:context_start]
        context_tokens = tokens[context_start + 2: -1]

        q_ents = self.pos_tags(question_tokens)
        c_ents = self.pos_tags(context_tokens)
        for k,v in q_ents.items():
            feat.add_new(f"_POS_{k}_Q", v)
        for k,v in c_ents.items():
            feat.add_new(f"_POS_{k}_C", v)
        return feat

    def extract_entity_features(self, tokens, ans_range):
        feat = common.IndexedFeature()
        context_start = tokens.index(self.sep_tokens[1])
        question_tokens = tokens[1:context_start]
        context_tokens = tokens[context_start + 2: -1]

        q_ents = self.endf(question_tokens, source="Q")
        c_ents = self.endf(context_tokens, source="C")
        for f in q_ents:
            feat.add_new(f[0], f[1])
        for f in c_ents:
            feat.add_new(f[0], f[1])
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
        # print(np.array(list(attr["attributions"])))
        # attri = [i[1] for i in attr["attributions"]]
        # print(attri)
        # attribution_val = np.array(attr["attributions"])
        attribution_val = attr['attribution'].numpy().copy()
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

    def extract_state_features(self, attr, tokens, ans_range, states):
        feat = common.IndexedFeature()
        context_start = tokens.index(self.sep_tokens[1])
        # print(states.shape)
        # rep = torch.mean(states, dim=-1)
        # repr = rep.tolist()[0]
        # question_repr = repr[1:context_start]
        # context_repr = repr[context_start + 2: -1]
        # print(torch.mean(states, dim=1).shape)
        # mean_token_states = torch.mean(states, dim=1)
        start, end = ans_range
        ans_indices = list(range(start, end + 1))
        # print(ans_indices)
        # print(states.shape)
        q_states = states[:, 1:context_start, :]
        # print(q_states.shape)
        # print(q_states)
        c_states = states[:, context_start + 2: -1, :]
        torch.set_printoptions(threshold=10_000)
        # print(c_states)
        # print(c_states[:, :start-(context_start+2), :])
        # remove answer states from context states
        c_states_before = c_states[:, :start-(context_start+2), :]
        c_states_after = c_states[:, end+1-(context_start+2):, :]
        final_c_states = torch.cat((c_states_before, c_states_after), dim=1)
        # print(final_c_states.shape)

        a_states = states[:, start:end+1, :]
        # print(a_states)

        attribution = np.array(attr["attribution"])
        # print(len(attribution))
        question_attr = attribution[1:context_start]
        # print(len(question_attr))
        context_attr = attribution[context_start + 2: -1]
        c_attr_before = context_attr[:start - (context_start + 2)]
        c_attr_after = context_attr[end + 1 - (context_start + 2):]
        final_c_attr = np.concatenate((c_attr_before, c_attr_after), axis=0)
        # print(len(context_attr))
        ans_attr = attribution[start: end+1]
        # print(len(ans_attr))

        ques_sorted_indices = np.argsort(question_attr)
        # take topk indices
        topk_ques_indices = ques_sorted_indices[-int(ques_sorted_indices.shape[0]*0.10):]
        # print(ques_sorted_indices.shape)
        ques_topk_states = np.take(q_states, topk_ques_indices, axis=1) # .tolist()[0]

        topc_percent = 0.10
        cxt_sorted_indices = np.argsort(final_c_attr)
        cxt_top_k_indices = cxt_sorted_indices[-int(cxt_sorted_indices.shape[0] * topc_percent):]
        cxt_topk_states = np.take(final_c_states, cxt_top_k_indices, axis=1) # .tolist()[0]

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

    def extract_polarity_feature(self, attr, tags, words, tags_for_tok, ans_index, polarity, include_basic=True,
                                 include_stats=False):
        named_feat = common.IndexedFeature()
        token_attribution = self.aggregate_token_attribution(attr, tags, polarity)
        assert token_attribution.size == len(words)
        # if link_attribution is not None:
        #     assert link_attribution.shape == (len(words), len(words))
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

            # for entites TODO: change format later
            # named_feat.add_set(self.extract_token_attr_feature_in_question(
            #     words,
            #     ents_for_tok,
            #     token_attribution))
            # named_feat.add_set(self.extract_token_attr_feature_in_context(
            #     words,
            #     ents_for_tok,
            #     token_attribution))
            # named_feat.add_set(
            #     self.extract_token_attr_feature_in_input(
            #         words,
            #         ents_for_tok,
            #         token_attribution,
            #         ans_index))

            # if link_attribution is not None:
            #     named_feat.add_set(self.extract_link_attr_feature(tags_for_tok, link_attribution, ans_index))
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
        # print(self.tagger_info)
        # print('extracting features for instance', attr)
        # print(preds)
        pred_text = preds[0]['text']
        # print(pred_text)
        example = attr['example']
        # print("------------------")
        # print(preds)
        # print("example:", example)
        if self.dataset == "squad":
            gold_text = example[2]['text']
        else:
            gold_text = example[2]
        # print('gold text', gold_text)
        # print(gold_text)

        exact_match = evaluate.get_score(
            metric="exact_match",
            pred_text=pred_text,
            gold_text=gold_text
        )
        f1 = evaluate.get_score(metric="f1", pred_text=pred_text, gold_text=gold_text)
        calib_label = 1 if exact_match > 0 else 0
        print(calib_label)

        # baseline features
        named_feat = common.IndexedFeature()
        named_feat.add_set(self.extract_baseline_feature(preds, attr))

        # print("attr:", attr["prediction"])
        start_index, end_index = attr['prelim_result'][0], \
                                 attr['prelim_result'][1]
        # print("start_index:", start_index)
        # print("end_index:", end_index)
        segments = tags["segments"]

        transformed_start_idx = 0
        while segments[transformed_start_idx + 1][0] <= start_index:
            transformed_start_idx += 1
        transformed_end_idx = 0
        while segments[transformed_end_idx][1] < (end_index + 1):
            transformed_end_idx += 1

        ans_range = (transformed_start_idx, transformed_end_idx)
        # print("ans range:", ans_range)
        # syntactic features
        words, tags_for_tok = tags['words'], tags['tags']
        # words = [w for w in words if w != " "]
        tags_for_tok = [self.lemmatize_pos_tag(x) for x in tags_for_tok]

        named_feat.add_set(self.extract_state_features(attr, words, ans_range, states))
        named_feat.add_set(self.extract_bow_feature(words, tags_for_tok, ans_range))
        named_feat.add_set(self.extract_polarity_feature(attr, tags, words, tags_for_tok, ans_range, 'NEU'))
        # print({'feature': named_feat, 'label': calib_label, 'f1_score': f1})
        return {'feature': named_feat, 'label': calib_label, 'f1_score': f1}


if __name__ == "__main__":
    # base_model = "roberta-base"
    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = "gpt_neox"
    dataset = "hotpot_qa"
    method = "shap"
    data_path = f"./src/data/{dataset}"
    feat = CreateFeatures(path=data_path, model=model, dataset=dataset, method=method)
    feat.featurize()
