import ast
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

from src.calibration.baseline import common, evaluate, utils

import warnings

# ignore warning
warnings.filterwarnings("ignore")


class CreateFeatures:
    def __init__(self, path):
        self.path = path
        self.cls_tokens = ["[CLS]", "<s>"]
        self.sep_tokens = ["[SEP]", "</s>"]
        self.count = 0
        self.nlp = spacy.load("en_core_web_sm")
        # self.supar = Parser.load('crf-con-en')

    def _load_data(self):

        self.tagger_info = utils.load_bin(f"{self.path}/pos_info.bin")
        # self.attributions = utils.load_bin(f"{self.path}/ig_info_with_cf.bin")
        self.attributions = self.build_file_dict(
            dataset="squad", split="addsent-dev", method="shap"
        )
        pred_df = pd.read_csv(f"{self.path}/outputs.csv")
        self.preds_info = pred_df.set_index("id").T.to_dict("dict")
        # self.preds_info = utils.read_json('outputs/addsent-dev_squad_predictions.json')

    def load_interp_info(self, file_dict, qas_id):
        return torch.load(file_dict[qas_id])

    def build_file_dict(self, dataset, split, method):
        # hard-coded path here: be careful
        # prefix = 'squad_sample-addsent_roberta-base'
        prefix = "{}_{}_roberta".format(dataset, split)
        fnames = os.listdir(os.path.join("explanations", method, prefix))
        qa_ids = [x.split("-", 1)[1].split(".")[0] for x in fnames]
        fullnames = [os.path.join("explanations", method, prefix, x) for x in fnames]
        return dict(zip(qa_ids, fullnames))

    def featurize(self):
        self._load_data()
        processed_instances = OrderedDict()
        c = 0
        # print(self.attributions)
        for q_id in tqdm(
            self.tagger_info, total=len(self.tagger_info), desc="transforming..."
        ):
            # print(q_id)
            tags = self.tagger_info[q_id]
            # print(tags)
            interp = self.load_interp_info(self.attributions, q_id)
            # print(interp)
            attr_len = len(interp["attribution"])
            interp["attribution"] = torch.rand(attr_len)
            # print(len(ast.literal_eval(attributions)["attributions"]))
            preds = self.preds_info[q_id]
            # print(preds)
            # print(attributions)
            # print(tags)
            # print(len(tags["words"]))
            # print(len(ast.literal_eval(attributions)))
            # print(attributions)
            # print(ast.literal_eval(preds["pred_text"])[0])
            processed_instances[q_id] = self.extract_feature_for_instance(
                interp, tags, preds
            )
            c += 1
            # if c==2:
            #     break
            # print(self.count)
        # print(processed_instances)
        utils.dump_to_bin(processed_instances, f"{self.path}/calibration_data.bin")

    def lemmatize_pos_tag(self, token_tags):
        """
        limit the amount of POS tags to reduce the number of features
        :param token_tags: tagged tokens
        :return:
        """
        tok, pos, tag = token_tags
        if tag == "NNS":
            tag = "NN"
        if tag == "NNPS":
            tag = "NNP"
        if tag.startswith("JJ"):
            tag = "JJ"
        if tag.startswith("RB"):
            tag = "RB"
        if tag.startswith("W"):
            tag = "W"
        if tag.startswith("PRP"):
            tag = "PRP"
        if tag.startswith("VB"):
            tag = "VB"
        if pos == "PUNCT":  # or tag in [":"]:
            tag = "PUNCT"
        # if tag == '$':
        #     tag = 'SYM'
        # if tag == "``" or tag == "''":
        #     tag = "AUX"
        return tok, pos, tag

    def extract_baseline_feature(self, preds):
        # print(preds)
        predictions = ast.literal_eval(preds["pred_text"])[0]
        # print(predictions)
        feat = common.IndexedFeature()
        # base features
        for rank, p in enumerate(predictions[:5]):
            feat.add(f"BASELINE_PROB_{rank}", p["score"])
        feat.add("BASELINE_CONTEXT_LENGTH", len(preds["context"]))
        feat.add("BASELINE_PRED_ANS_LENGTH", len(predictions[0]["answer"]))

        top_pred = predictions[0]["answer"]
        first_distinct_prob = 0
        for i, p in enumerate(predictions[1:]):
            overlapping = evaluate.f1_score(p["answer"], top_pred)
            if overlapping > 0:
                continue
            first_distinct_prob = p["score"]
            # print(i+1, top_pred, p['answer'], first_distinct_prob)
        feat.add("FIRST_DISTINCT_PROB", first_distinct_prob)

        return feat

    def source_of_token(self, idx, tok, context_start, ans_range):
        if tok in self.cls_tokens or tok in self.sep_tokens:
            return "S"
        if idx >= 1 and idx < context_start:
            return "Q"
        if idx >= ans_range[0] and idx <= ans_range[1]:
            # print(tok)
            return "A"
        return "C"

    def nan_check(self, result):
        for i, (key, value) in enumerate(result):
            if math.isnan(float(value)):
                result[i][1] = 0
        return result

    def count_syllables(self, word: str):
        return len(
            re.findall("(?!e$)[aeiouy]+", word, re.I)
            + re.findall("^[^aeiouy]*e$", word, re.I)
        )

    def handle_zero_division(self, n, d):
        return n / d if d else 0

    def worF(self, tokens):
        words = [x.lstrip() for x in tokens]
        spaces = [
            False if i == len(tokens) - 1 else tokens[i + 1][0] == " "
            for i in range(len(tokens))
        ]

        valid_idx = [i for i, w in enumerate(words) if len(w)]
        words = [words[i] for i in valid_idx]
        spaces = [spaces[i] for i in valid_idx]
        doc = Doc(self.nlp.vocab, words=words, spaces=spaces)
        processed_tokens = self.nlp(doc)
        n_sent, n_token = 0, 0
        token_list = []
        for sent in processed_tokens.sents:
            n_sent += 1
            for token in sent:
                token_list.append(token.lemma_.lower())
                n_token += 1

        to_SbFrQ_C = 0
        to_SbCDC_C = 0
        to_SbFrL_C = 0
        to_SbCDL_C = 0
        to_SbSBW_C = 0
        to_SbL1W_C = 0
        to_SbSBC_C = 0
        to_SbL1C_C = 0

        DB = pd.read_csv("./src/resources/SUBTLEXus.csv")
        DB.set_index("Word_lowercased", inplace=True, drop=True)
        for token in token_list:
            if token in DB.index:
                scores_for_this_token = list(DB.loc[token, :])
                for i, score in enumerate(scores_for_this_token):
                    scores_for_this_token[i] = (
                        0 if str(score) == "none" else scores_for_this_token[i]
                    )
                to_SbFrQ_C += float(scores_for_this_token[1])
                to_SbCDC_C += float(scores_for_this_token[2])
                to_SbFrL_C += float(scores_for_this_token[3])
                to_SbCDL_C += float(scores_for_this_token[4])
                to_SbSBW_C += float(scores_for_this_token[5])
                to_SbL1W_C += float(scores_for_this_token[6])
                to_SbSBC_C += float(scores_for_this_token[7])
                to_SbL1C_C += float(scores_for_this_token[8])

        result = {
            "to_SbFrQ_C": to_SbFrQ_C,
            "as_SbFrQ_C": to_SbFrQ_C / n_sent,
            "at_SbFrQ_C": to_SbFrQ_C / n_token,
            "to_SbCDC_C": to_SbCDC_C,
            "as_SbCDC_C": to_SbCDC_C / n_sent,
            "at_SbCDC_C": to_SbCDC_C / n_token,
            "to_SbFrL_C": to_SbFrL_C,
            "as_SbFrL_C": to_SbFrL_C / n_sent,
            "at_SbFrL_C": to_SbFrL_C / n_token,
            "to_SbCDL_C": to_SbCDL_C,
            "as_SbCDL_C": to_SbCDL_C / n_sent,
            "at_SbCDL_C": to_SbCDL_C / n_token,
            "to_SbSBW_C": to_SbSBW_C,
            "as_SbSBW_C": to_SbSBW_C / n_sent,
            "at_SbSBW_C": to_SbSBW_C / n_token,
            "to_SbL1W_C": to_SbL1W_C,
            "as_SbL1W_C": to_SbL1W_C / n_sent,
            "at_SbL1W_C": to_SbL1W_C / n_token,
            "to_SbSBC_C": to_SbSBC_C,
            "as_SbSBC_C": to_SbSBC_C / n_sent,
            "at_SbSBC_C": to_SbSBC_C / n_token,
            "to_SbL1C_C": to_SbL1C_C,
            "as_SbL1C_C": to_SbL1C_C / n_sent,
            "at_SbL1C_C": to_SbL1C_C / n_token,
        }
        return result

    def extract_worf_features(self, tokens, ans_range):
        feat = common.IndexedFeature()
        context_start = tokens.index(self.sep_tokens[1])
        question_tokens = tokens[1:context_start]
        context_tokens = tokens[context_start + 2 : -1]

        q_ents = self.worF(question_tokens)
        c_ents = self.worF(context_tokens)
        # print(q_ents)
        # print(c_ents)
        for k, v in q_ents.items():
            feat.add_new(f"WOR_{k}_Q", v)
        for k, v in c_ents.items():
            feat.add_new(f"WOR_{k}_C", v)
        return feat

    def ttrF(self, tokens):
        words = [x.lstrip() for x in tokens]
        spaces = [
            False if i == len(tokens) - 1 else tokens[i + 1][0] == " "
            for i in range(len(tokens))
        ]

        valid_idx = [i for i, w in enumerate(words) if len(w)]
        words = [words[i] for i in valid_idx]
        spaces = [spaces[i] for i in valid_idx]
        doc = Doc(self.nlp.vocab, words=words, spaces=spaces)
        processed_tokens = self.nlp(doc)
        n_sent, n_token = 0, 0
        token_list = []
        for sent in processed_tokens.sents:
            n_sent += 1
            for token in sent:
                token_list.append(token.lemma_.lower())
                n_token += 1

        n_utoken = 1
        default_MTLD = 0.72
        MTLD_count = 0
        for token in token_list:
            if token_list.count(token) == 1:
                n_utoken += 1
            if float(n_utoken / n_token) >= 0.72:
                MTLD_count += 1

        result = {
            "SimpTTR_S": float(n_utoken / n_token),
            "CorrTTR_S": float(n_utoken / math.sqrt(2 * n_token)),
            "BiLoTTR_S": float(math.log(n_utoken) / math.log(n_token)),
            "UberTTR_S": float(
                self.handle_zero_division(
                    ((math.log(n_utoken)) ** 2), (math.log(n_token / n_utoken))
                )
            ),
            "MTLDTTR_S": float(MTLD_count),
        }
        return result

    def extract_ttrf_features(self, tokens, ans_range):
        feat = common.IndexedFeature()
        context_start = tokens.index(self.sep_tokens[1])
        question_tokens = tokens[1:context_start]
        context_tokens = tokens[context_start + 2 : -1]

        q_ents = self.ttrF(question_tokens)
        c_ents = self.ttrF(context_tokens)
        # print(q_ents)
        # print(c_ents)
        for k, v in q_ents.items():
            feat.add_new(f"TTR_{k}_Q", v)
        for k, v in c_ents.items():
            feat.add_new(f"TTR_{k}_C", v)
        return feat

    def varF(self, tokens):

        words = [x.lstrip() for x in tokens]
        spaces = [
            False if i == len(tokens) - 1 else tokens[i + 1][0] == " "
            for i in range(len(tokens))
        ]

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

        noun_list = []
        verb_list = []
        adje_list = []
        adve_list = []
        n_unoun = 0
        n_uverb = 0
        n_uadje = 0
        n_uadve = 0

        for token in processed_tokens:
            if token.pos_ == "NOUN":
                noun_list.append(token.lemma)
            if token.pos_ == "VERB":
                verb_list.append(token.lemma)
            if token.pos_ == "ADJ":
                adje_list.append(token.lemma)
            if token.pos_ == "ADV":
                adve_list.append(token.lemma)
        for noun in noun_list:
            if noun_list.count(noun) == 1:
                n_unoun += 1
        for verb in verb_list:
            if verb_list.count(verb) == 1:
                n_uverb += 1
        for adje in adje_list:
            if adje_list.count(adje) == 1:
                n_uadje += 1
        for adve in adve_list:
            if adve_list.count(adve) == 1:
                n_uadve += 1

        result = {
            "SimpNoV_S": float(self.handle_zero_division(n_unoun, (len(noun_list)))),
            "SquaNoV_S": float(
                self.handle_zero_division((n_unoun) ** 2, (len(noun_list)))
            ),
            "CorrNoV_S": float(
                self.handle_zero_division(n_unoun, (math.sqrt(2 * len(noun_list))))
            ),
            "SimpVeV_S": float(self.handle_zero_division(n_uverb, (len(verb_list)))),
            "SquaVeV_S": float(
                self.handle_zero_division((n_uverb) ** 2, (len(verb_list)))
            ),
            "CorrVeV_S": float(
                self.handle_zero_division(n_uverb, (math.sqrt(2 * len(verb_list))))
            ),
            "SimpAjV_S": float(self.handle_zero_division(n_uadje, (len(adje_list)))),
            "SquaAjV_S": float(
                self.handle_zero_division((n_uadje) ** 2, (len(adje_list)))
            ),
            "CorrAjV_S": float(
                self.handle_zero_division(n_uadje, (math.sqrt(2 * len(adje_list))))
            ),
            "SimpAvV_S": float(self.handle_zero_division(n_uadve, (len(adve_list)))),
            "SquaAvV_S": float(
                self.handle_zero_division((n_uadve) ** 2, (len(adve_list)))
            ),
            "CorrAvV_S": float(
                self.handle_zero_division(n_uadve, (math.sqrt(2 * len(adve_list))))
            ),
        }

        return result

    def extract_varf_features(self, tokens, ans_range):
        feat = common.IndexedFeature()
        context_start = tokens.index(self.sep_tokens[1])
        question_tokens = tokens[1:context_start]
        context_tokens = tokens[context_start + 2 : -1]

        q_ents = self.varF(question_tokens)
        c_ents = self.varF(context_tokens)
        # print(q_ents)
        # print(c_ents)
        for k, v in q_ents.items():
            feat.add_new(f"VAR_{k}_Q", v)
        for k, v in c_ents.items():
            feat.add_new(f"VAR_{k}_C", v)
        return feat

    def pos_tags(self, tokens):
        words = [x.lstrip() for x in tokens]
        spaces = [
            False if i == len(tokens) - 1 else tokens[i + 1][0] == " "
            for i in range(len(tokens))
        ]

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
            if (
                token.pos_ == "NOUN"
                or token.pos_ == "VERB"
                or token.pos_ == "NUM"
                or token.pos_ == "ADJ"
                or token.pos_ == "ADV"
            ):
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
        context_tokens = tokens[context_start + 2 : -1]

        q_ents = self.pos_tags(question_tokens)
        c_ents = self.pos_tags(context_tokens)
        # print(q_ents)
        # print(c_ents)
        for k, v in q_ents.items():
            feat.add_new(f"_POS_{k}_Q", v)
        for k, v in c_ents.items():
            feat.add_new(f"_POS_{k}_C", v)
        return feat

    def shaf(self, tokens, source):
        words = [x.lstrip() for x in tokens]
        spaces = [
            False if i == len(tokens) - 1 else tokens[i + 1][0] == " "
            for i in range(len(tokens))
        ]

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

        total_count_char = len(words)
        total_count_tokn = n_token
        total_count_syll = 0
        for token in words:
            total_count_syll += self.count_syllables(token)
        result = [
            (f"SHA_TokSenM_S_{source}", float(n_token * n_sent)),
            (f"SHA_TokSenS_S_{source}", float(math.sqrt(n_token * n_sent))),
            (
                f"SHA_TokSenL_S_{source}",
                float(self.handle_zero_division(math.log(n_token), math.log(n_sent))),
            ),
            (
                f"SHA_as_Token_C_{source}",
                float(self.handle_zero_division(total_count_tokn, n_sent)),
            ),
            (
                f"SHA_as_Sylla_C_{source}",
                float(self.handle_zero_division(total_count_syll, n_sent)),
            ),
            (f"SHA_at_Sylla_C_{source}", float(total_count_syll / n_token)),
            (
                f"SHA_as_Chara_C_{source}",
                float(self.handle_zero_division(total_count_char, n_sent)),
            ),
            (f"SHA_at_Chara_C_{source}", float(total_count_char / n_token)),
        ]
        # result = self.nan_check(result)
        return result

    def psyF(self, tokens, source):
        words = [x.lstrip() for x in tokens]
        spaces = [
            False if i == len(tokens) - 1 else tokens[i + 1][0] == " "
            for i in range(len(tokens))
        ]

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

        to_AAKuW_C = 0
        to_AAKuL_C = 0
        to_AABiL_C = 0
        to_AABrL_C = 0
        to_AACoL_C = 0

        DB = pd.read_csv("./src/resources/AoAKuperman.csv")
        DB.set_index("Word", inplace=True, drop=True)
        for token in words:
            if token in DB.index:
                scores_for_this_token = list(DB.loc[token, :])
                for i, score in enumerate(scores_for_this_token):
                    scores_for_this_token[i] = (
                        0 if str(score) == "none" else scores_for_this_token[i]
                    )
                to_AAKuW_C += float(scores_for_this_token[7])
                to_AAKuL_C += float(scores_for_this_token[9])
                to_AABiL_C += float(scores_for_this_token[11])
                to_AABrL_C += float(scores_for_this_token[12])
                to_AACoL_C += float(scores_for_this_token[13])

        result = [
            (f"PSY_to_AAKuW_C_{source}", to_AAKuW_C),
            (f"PSY_as_AAKuW_C_{source}", to_AAKuW_C / n_sent),
            (f"PSY_at_AAKuW_C_{source}", to_AAKuW_C / n_token),
            (f"PSY_to_AAKuL_C_{source}", to_AAKuL_C),
            (f"PSY_as_AAKuL_C_{source}", to_AAKuL_C / n_sent),
            (f"PSY_at_AAKuL_C_{source}", to_AAKuL_C / n_token),
            (f"PSY_to_AABiL_C_{source}", to_AABiL_C),
            (f"PSY_as_AABiL_C_{source}", to_AABiL_C / n_sent),
            (f"PSY_at_AABiL_C_{source}", to_AABiL_C / n_token),
            (f"PSY_to_AABrL_C_{source}", to_AABrL_C),
            (f"PSY_as_AABrL_C_{source}", to_AABrL_C / n_sent),
            (f"PSY_at_AABrL_C_{source}", to_AABrL_C / n_token),
            (f"PSY_to_AACoL_C_{source}", to_AABrL_C),
            (f"PSY_as_AACoL_C_{source}", to_AABrL_C / n_sent),
            (f"PSY_at_AACoL_C_{source}", to_AABrL_C / n_token),
        ]
        return result

    def endf(self, tokens, source):

        words = [x.lstrip() for x in tokens]
        spaces = [
            False if i == len(tokens) - 1 else tokens[i + 1][0] == " "
            for i in range(len(tokens))
        ]

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

        to_EntiM_C = 0
        to_UEnti_C = 0
        ent_list = []
        unique_ent_list = []

        for ent in processed_tokens.ents:
            to_EntiM_C += 1
            ent_list.append(ent.text)

        for ent in ent_list:
            if ent_list.count(ent) == 1:
                to_UEnti_C += 1
                unique_ent_list.append(ent)

        result = [
            # total number of Entities Mentions counts
            (f"ENT_to_EntiM_{source}", to_EntiM_C),
            # average number of Entities Mentions counts per sentence
            (f"ENT_as_EntiM_{source}", to_EntiM_C / n_sent),
            # average number of Entities Mentions counts per token (word)
            (f"ENT_at_EntiM_{source}", to_EntiM_C / n_token),
            # unique ents...
            (f"ENT_to_UEnti_{source}", to_UEnti_C),
            (f"ENT_as_UEnti_{source}", to_UEnti_C / n_sent),
            (f"ENT_at_UEnti_{source}", to_UEnti_C / n_token),
        ]

        return result

    def extract_entity_features(self, tokens, ans_range):
        feat = common.IndexedFeature()
        context_start = tokens.index(self.sep_tokens[1])
        question_tokens = tokens[1:context_start]
        context_tokens = tokens[context_start + 2 : -1]

        q_ents = self.endf(question_tokens, source="Q")
        c_ents = self.endf(context_tokens, source="C")
        # print(q_ents)
        # print(c_ents)
        for f in q_ents:
            feat.add_new(f[0], f[1])
        for f in c_ents:
            feat.add_new(f[0], f[1])
        return feat

    def extract_shallow_features(self, tokens, ans_range):
        feat = common.IndexedFeature()
        context_start = tokens.index(self.sep_tokens[1])
        question_tokens = tokens[1:context_start]
        context_tokens = tokens[context_start + 2 : -1]

        q_ents = self.shaf(question_tokens, source="Q")
        c_ents = self.shaf(context_tokens, source="C")
        # print(q_ents)
        # print(c_ents)
        for f in q_ents:
            feat.add_new(f[0], f[1])
        for f in c_ents:
            feat.add_new(f[0], f[1])
        return feat

    def extract_psy_features(self, tokens, ans_range):
        feat = common.IndexedFeature()
        context_start = tokens.index(self.sep_tokens[1])
        question_tokens = tokens[1:context_start]
        context_tokens = tokens[context_start + 2 : -1]

        q_ents = self.psyF(question_tokens, source="Q")
        c_ents = self.psyF(context_tokens, source="C")
        # print(q_ents)
        # print(c_ents)
        for f in q_ents:
            feat.add_new(f[0], f[1])
        for f in c_ents:
            feat.add_new(f[0], f[1])
        return feat

    def extract_bow_feature(self, words, tags, ans_range):
        feat = common.IndexedFeature()
        context_start = words.index(self.sep_tokens[1])
        # print(words)
        # print(words[ans_range[0]: ans_range[1]])
        # # print(words[ans_range[1]])
        # print(context_start)
        # print(ans_range)
        for i, (i_token, i_pos, i_tag) in enumerate(tags):
            i_src = self.source_of_token(i, i_token, context_start, ans_range)
            if i_src == "Q" or i_src == "A" or i_src == "C":
                # print('BOW_{}_{}'.format(i_src, i_tag))
                feat.add("BOW_{}_{}".format(i_src, i_tag))
                feat.add("BOW_IN_{}".format(i_tag))

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
        attribution_val = attr["attribution"].numpy().copy()
        if polarity == "POS":
            attribution_val[attribution_val < 0] = 0
        elif polarity == "NEG":
            attribution_val[attribution_val > 0] = 0
        elif polarity == "NEU":
            pass
        else:
            raise RuntimeError("Invalid polarity")

        attribution_val = self.merge_attribution_by_segments(
            attribution_val, tags["segments"]
        )
        assert attribution_val.shape[0] == len(tags["segments"])
        # normalize
        attribution_val = attribution_val / np.sum(attribution_val)
        # print(attribution_val)
        return attribution_val

    def normalize_token_attr(self, feat, attributions, norm_method=None):
        if norm_method is None:
            return feat
        if norm_method == "all":
            sum_v = np.sum(attributions)
            for k in feat.data:
                feat.data[k] = feat.data[k] / sum_v if sum_v != 0 else 0
            return feat
        if norm_method == "counted":
            sum_v = sum(feat.data.values())
            for k in feat.data:
                feat.data[k] = feat.data[k] / sum_v if sum_v != 0 else 0
            return feat
        raise RuntimeError(norm_method)

    def extract_token_attr_feature_in_question(
        self, words, tags, attributions, include_punct=False
    ):
        context_start = words.index(self.sep_tokens[1])
        tags = tags[1:context_start]
        attributions = attributions[1:context_start]

        feat = common.IndexedFeature()
        unnorm = common.IndexedFeature()
        sum_v = 0
        for i, (token, pos, tag) in enumerate(tags):
            if tag:
                v = attributions[i]
                if pos == "PUNCT" and not include_punct:
                    continue
                feat.add("NORMED_TOK_Q_" + tag, v)
                unnorm.add("UNNORM_TOK_Q_" + tag, v)
                sum_v += v

        feat = self.normalize_token_attr(feat, attributions)
        feat.add_set(unnorm)
        feat.add("SUM_TOK_Q", sum_v)
        return feat

    def extract_token_attr_feature_in_context(
        self, words, tags, attributions, include_punct=False
    ):
        context_start = words.index(self.sep_tokens[1])
        tags = tags[context_start + 2 : -1]
        attributions = attributions[context_start + 2 : -1]

        feat = common.IndexedFeature()
        unnorm = common.IndexedFeature()
        sum_v = 0
        for i, (token, pos, tag) in enumerate(tags):
            if tag:
                v = attributions[i]
                if pos == "PUNCT" and not include_punct:
                    continue
                feat.add("NORMED_TOK_C_" + tag, v)
                feat.add("UNNORM_TOK_C_" + tag, v)
                sum_v += v

        feat = self.normalize_token_attr(feat, attributions)
        feat.add_set(unnorm)
        feat.add("SUM_TOK_C", sum_v)
        return feat

    def extract_token_attr_feature_in_input(
        self, words, tags, attributions, ans_range, include_punct=False
    ):
        feat = common.IndexedFeature()
        context_start = words.index(self.sep_tokens[1])
        normed_a_feat = common.IndexedFeature()
        unnormed_a_feat = common.IndexedFeature()
        for i, (token, pos, tag) in enumerate(tags):
            if tag:
                v = attributions[i]
                if pos == "PUNCT" and not include_punct:
                    continue
                feat.add("TOK_IN_" + tag, v)

                if i >= ans_range[0] and i <= ans_range[1]:
                    unnormed_a_feat.add("UNNORM_TOK_A_" + tag, v)
                    normed_a_feat.add("NORMED_TOK_A_" + tag, v)

        feat.add_set(self.normalize_token_attr(normed_a_feat, [], "counted"))
        feat.add_set(unnormed_a_feat)
        return feat

    def extract_token_attr_stats_in_input(self, words, tags, attributions, part):
        feat = common.IndexedFeature()
        context_start = words.index(self.sep_tokens[1])
        if part == "Q":
            tags = tags[1:context_start]
            attributions = attributions[1:context_start]
        if part == "C":
            tags = tags[context_start + 2 : -1]
            attributions = attributions[context_start + 2 : -1]

        feat.add("STAT_MEAN_" + part, attributions.mean())
        feat.add("STAT_STD_" + part, attributions.std())
        return feat

    def extract_polarity_feature(
        self,
        attr,
        tags,
        words,
        tags_for_tok,
        ans_index,
        polarity,
        include_basic=True,
        include_stats=False,
    ):
        named_feat = common.IndexedFeature()
        token_attribution = self.aggregate_token_attribution(attr, tags, polarity)
        # link_attribution = None
        # elif args.method in ['probe']:
        # link_attribution = self.aggregate_link_attribution(attr, polarity)
        # print(token_attribution.size, len(words))
        # print(words)
        # print(attr)
        #
        # if token_attribution.size != len(words):
        #     print(token_attribution.size, len(words))
        #     print(words)
        #     self.count += 1
        # assert token_attribution.size == len(words)
        # if link_attribution is not None:
        #     assert link_attribution.shape == (len(words), len(words))
        if include_basic:
            named_feat.add_set(
                self.extract_token_attr_feature_in_question(
                    words, tags_for_tok, token_attribution
                )
            )
            named_feat.add_set(
                self.extract_token_attr_feature_in_context(
                    words, tags_for_tok, token_attribution
                )
            )
            named_feat.add_set(
                self.extract_token_attr_feature_in_input(
                    words, tags_for_tok, token_attribution, ans_index
                )
            )

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
            named_feat.add_set(
                self.extract_token_attr_stats_in_input(
                    words, tags_for_tok, token_attribution, "Q"
                )
            )
            named_feat.add_set(
                self.extract_token_attr_stats_in_input(
                    words, tags_for_tok, token_attribution, "C"
                )
            )
            named_feat.add_set(
                self.extract_token_attr_stats_in_input(
                    words, tags_for_tok, token_attribution, "IN"
                )
            )
        named_feat.add_prefix(polarity + "_")
        return named_feat

    def extract_feature_for_instance(self, attr, tags, preds):
        # print(self.tagger_info)
        # print(preds)
        predictions = ast.literal_eval(preds["pred_text"])[0]
        # print(predictions)
        # create labels
        pred_text = predictions[0]["answer"]
        # print(pred_text)
        gold_text = ast.literal_eval(preds["gold_text"])  # preds['gold_text']
        # print(gold_text)

        exact_match = evaluate.get_score(
            metric="exact_match", pred_text=pred_text, gold_text=gold_text
        )
        f1 = evaluate.get_score(metric="f1", pred_text=pred_text, gold_text=gold_text)
        calib_label = 1 if exact_match > 0 else 0
        print(calib_label)

        # baseline features
        named_feat = common.IndexedFeature()
        named_feat.add_set(self.extract_baseline_feature(preds))

        start_index, end_index = (
            ast.literal_eval(preds["answer_start"])[0],
            ast.literal_eval(preds["answer_end"])[0],
        )
        segments = tags["segments"]
        transformed_start_idx = 0
        while segments[transformed_start_idx + 1][0] <= start_index:
            transformed_start_idx += 1
        transformed_end_idx = 0
        while segments[transformed_end_idx][1] < (end_index + 1):
            transformed_end_idx += 1
        ans_range = (transformed_start_idx, transformed_end_idx)
        # syntactic features
        words, tags_for_tok = tags["words"], tags["tags"]
        # words = [w for w in words if w != " "]
        tags_for_tok = [self.lemmatize_pos_tag(x) for x in tags_for_tok]

        # named_feat.add_set(self.extract_entity_features(words, ans_range))
        # named_feat.add_set(self.extract_shallow_features(words, ans_range))
        # named_feat.add_set(self.extract_psy_features(words, ans_range))
        # named_feat.add_set(self.extract_varf_features(words, ans_range))
        # named_feat.add_set(self.extract_ttrf_features(words, ans_range))
        # named_feat.add_set(self.extract_worf_features(words, ans_range))
        named_feat.add_set(self.extract_bow_feature(words, tags_for_tok, ans_range))
        named_feat.add_set(
            self.extract_polarity_feature(
                attr, tags, words, tags_for_tok, ans_range, "POS"
            )
        )
        # print({'feature': named_feat, 'label': calib_label, 'f1_score': f1})
        return {"feature": named_feat, "label": calib_label, "f1_score": f1}


if __name__ == "__main__":
    # base_model = "roberta-base"
    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    data_path = "./src/data/squad_adv_new"
    feat = CreateFeatures(path=data_path)
    feat.featurize()
