import string
import pandas as pd
from typing import List, Tuple
from collections import OrderedDict

import torch
import spacy
from spacy.tokens import Doc
from tqdm import tqdm
from supar import Parser

from transformers import (
    AutoTokenizer,
    # RobertaAdapterModel,
)

from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from src.calibration.baseline import dataloader


class FeatureExtractor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.eos_token
        self.supar = Parser.load("crf-con-en")

    def encode(
        self, inputs: list = None, add_special_tokens: bool = True, return_tensors=None
    ):
        """
        Encode inputs using the model tokenizer
        Args:
            inputs: question, context pair as a list
            add_special_tokens: where to add CLS, SEP tokens
            return_tensors: whether to return tensors
        Return:
            tokenized inputs
        """
        return self.tokenizer(
            inputs,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
            max_length=512,
        )

    def decode(self, input_ids: torch.Tensor, skip_special_tokens: bool) -> List[str]:
        """
        Decode received input_ids into a list of word tokens.
        Args:
            input_ids (torch.Tensor): Input ids representing
            word tokens for a sentence/document.
        """
        return self.tokenizer.convert_ids_to_tokens(
            input_ids[0], skip_special_tokens=skip_special_tokens
        )

    def _bpe_decode(self, tokens: List[str],) -> Tuple[List[str], List]:

        byte_encoder = bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}
        decoded_each_tok = [
            bytearray([byte_decoder[c] for c in t]).decode(
                encoding="utf-8", errors="replace"
            )
            for t in tokens
        ]

        end_points = []
        force_break = False
        for idx, token in enumerate(decoded_each_tok):
            # special token, punctuation, alphanumeric
            if (
                token in self.tokenizer.all_special_tokens
                or token in string.punctuation
                or not any([x.isalnum() for x in token.lstrip()])
                or token.lstrip == "'s"
            ):
                end_points.append(idx)
                force_break = True
                continue

            if force_break:
                end_points.append(idx)
                force_break = False
                continue

            if token[0] == " ":
                tokens[idx] = token[:]
                end_points.append(idx)

        end_points.append(len(tokens))

        segments = []
        for i in range(1, len(end_points)):
            if end_points[i - 1] == end_points[i]:
                continue
            segments.append((end_points[i - 1], end_points[i]))

        filtered_tokens = []
        for s0, s1 in segments:
            filtered_tokens.append("".join(decoded_each_tok[s0:s1]))

        return filtered_tokens, segments

    def process_input(self, request):
        inputs = [[request["question"], request["context"]]]
        encoded_inputs = self.encode(
            inputs, add_special_tokens=True, return_tensors="pt"
        )

        decoded_text = self.decode(
            encoded_inputs["input_ids"], skip_special_tokens=False
        )
        filtered_tokens, segments = self._bpe_decode(decoded_text)
        return filtered_tokens, segments

    def entity_features(self, tokens, nlp):
        words = [x.lstrip() for x in tokens]
        spaces = [
            False if i == len(tokens) - 1 else tokens[i + 1][0] == " "
            for i in range(len(tokens))
        ]

        valid_idx = [i for i, w in enumerate(words) if len(w)]
        words = [words[i] for i in valid_idx]
        spaces = [spaces[i] for i in valid_idx]
        doc = Doc(nlp.vocab, words=words, spaces=spaces)
        processed_tokens = nlp(doc)
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

        result = {
            # total number of Entities Mentions counts
            "to_EntiM_C": to_EntiM_C,
            # average number of Entities Mentions counts per sentence
            "as_EntiM_C": to_EntiM_C / n_sent,
            # average number of Entities Mentions counts per token (word)
            "at_EntiM_C": to_EntiM_C / n_token,
            # unique ents...
            "to_UEnti_C": to_UEnti_C,
            "as_UEnti_C": to_UEnti_C / n_sent,
            "at_UEnti_C": to_UEnti_C / n_token,
        }
        print(result)

    def retrieve(self, tokens, nlp):

        words = [x.lstrip() for x in tokens]
        spaces = [
            False if i == len(tokens) - 1 else tokens[i + 1][0] == " "
            for i in range(len(tokens))
        ]

        valid_idx = [i for i, w in enumerate(words) if len(w)]
        words = [words[i] for i in valid_idx]
        spaces = [spaces[i] for i in valid_idx]
        doc = Doc(nlp.vocab, words=words, spaces=spaces)
        processed_tokens = nlp(doc)
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

        result = {
            "to_AAKuW_C": to_AAKuW_C,
            "as_AAKuW_C": to_AAKuW_C / n_sent,
            "at_AAKuW_C": to_AAKuW_C / n_token,
            "to_AAKuL_C": to_AAKuL_C,
            "as_AAKuL_C": to_AAKuL_C / n_sent,
            "at_AAKuL_C": to_AAKuL_C / n_token,
            "to_AABiL_C": to_AABiL_C,
            "as_AABiL_C": to_AABiL_C / n_sent,
            "at_AABiL_C": to_AABiL_C / n_token,
            "to_AABrL_C": to_AABrL_C,
            "as_AABrL_C": to_AABrL_C / n_sent,
            "at_AABrL_C": to_AABrL_C / n_token,
            "to_AACoL_C": to_AABrL_C,
            "as_AACoL_C": to_AABrL_C / n_sent,
            "at_AACoL_C": to_AABrL_C / n_token,
        }
        return result

    def handle_zero_division(self, n, d):
        return n / d if d else 0

    def phrF(self, tokens, nlp):

        words = [x.lstrip() for x in tokens]
        spaces = [
            False if i == len(tokens) - 1 else tokens[i + 1][0] == " "
            for i in range(len(tokens))
        ]

        valid_idx = [i for i, w in enumerate(words) if len(w)]
        words = [words[i] for i in valid_idx]
        spaces = [spaces[i] for i in valid_idx]
        doc = Doc(nlp.vocab, words=words, spaces=spaces)
        processed_tokens = nlp(doc)
        n_sent, n_token = 0, 0
        sent_token_list = []
        for sent in processed_tokens.sents:
            n_sent += 1
            temp_list = []
            for token in sent:
                n_token += 1
                temp_list.append(token.text)
            sent_token_list.append(temp_list)

        to_NoPhr_C = 0
        to_VePhr_C = 0
        to_SuPhr_C = 0
        to_PrPhr_C = 0
        to_AjPhr_C = 0
        to_AvPhr_C = 0
        for sent in sent_token_list:
            dataset = self.supar.predict([sent], prob=True, verbose=False)
            parsed_tree = str(dataset.sentences)
            to_NoPhr_C += parsed_tree.count("NP")
            to_VePhr_C += parsed_tree.count("VP")
            to_SuPhr_C += parsed_tree.count("SBAR")
            to_PrPhr_C += parsed_tree.count("PP")
            to_AjPhr_C += parsed_tree.count("ADJP")
            to_AvPhr_C += parsed_tree.count("ADVP")
        result = {
            "to_NoPhr_C": to_NoPhr_C,
            "as_NoPhr_C": float(self.handle_zero_division(to_NoPhr_C, n_sent)),
            "at_NoPhr_C": float(self.handle_zero_division(to_NoPhr_C, n_token)),
            "ra_NoVeP_C": float(self.handle_zero_division(to_NoPhr_C, to_VePhr_C)),
            "ra_NoSuP_C": float(self.handle_zero_division(to_NoPhr_C, to_SuPhr_C)),
            "ra_NoPrP_C": float(self.handle_zero_division(to_NoPhr_C, to_PrPhr_C)),
            "ra_NoAjP_C": float(self.handle_zero_division(to_NoPhr_C, to_AjPhr_C)),
            "ra_NoAvP_C": float(self.handle_zero_division(to_NoPhr_C, to_AvPhr_C)),
            "to_VePhr_C": to_VePhr_C,
            "as_VePhr_C": float(self.handle_zero_division(to_VePhr_C, n_sent)),
            "at_VePhr_C": float(self.handle_zero_division(to_VePhr_C, n_token)),
            "ra_VeNoP_C": float(self.handle_zero_division(to_VePhr_C, to_NoPhr_C)),
            "ra_VeSuP_C": float(self.handle_zero_division(to_VePhr_C, to_SuPhr_C)),
            "ra_VePrP_C": float(self.handle_zero_division(to_VePhr_C, to_PrPhr_C)),
            "ra_VeAjP_C": float(self.handle_zero_division(to_VePhr_C, to_AjPhr_C)),
            "ra_VeAvP_C": float(self.handle_zero_division(to_VePhr_C, to_AvPhr_C)),
            "to_SuPhr_C": to_SuPhr_C,
            "as_SuPhr_C": float(self.handle_zero_division(to_SuPhr_C, n_sent)),
            "at_SuPhr_C": float(self.handle_zero_division(to_SuPhr_C, n_token)),
            "ra_SuNoP_C": float(self.handle_zero_division(to_SuPhr_C, to_NoPhr_C)),
            "ra_SuVeP_C": float(self.handle_zero_division(to_SuPhr_C, to_VePhr_C)),
            "ra_SuPrP_C": float(self.handle_zero_division(to_SuPhr_C, to_PrPhr_C)),
            "ra_SuAjP_C": float(self.handle_zero_division(to_SuPhr_C, to_AjPhr_C)),
            "ra_SuAvP_C": float(self.handle_zero_division(to_SuPhr_C, to_AvPhr_C)),
            "to_PrPhr_C": to_PrPhr_C,
            "as_PrPhr_C": float(self.handle_zero_division(to_PrPhr_C, n_sent)),
            "at_PrPhr_C": float(self.handle_zero_division(to_PrPhr_C, n_token)),
            "ra_PrNoP_C": float(self.handle_zero_division(to_PrPhr_C, to_NoPhr_C)),
            "ra_PrVeP_C": float(self.handle_zero_division(to_PrPhr_C, to_VePhr_C)),
            "ra_PrSuP_C": float(self.handle_zero_division(to_PrPhr_C, to_SuPhr_C)),
            "ra_PrAjP_C": float(self.handle_zero_division(to_PrPhr_C, to_AjPhr_C)),
            "ra_PrAvP_C": float(self.handle_zero_division(to_PrPhr_C, to_AvPhr_C)),
            "to_AjPhr_C": to_AjPhr_C,
            "as_AjPhr_C": float(self.handle_zero_division(to_AjPhr_C, n_sent)),
            "at_AjPhr_C": float(self.handle_zero_division(to_AjPhr_C, n_token)),
            "ra_AjNoP_C": float(self.handle_zero_division(to_AjPhr_C, to_NoPhr_C)),
            "ra_AjVeP_C": float(self.handle_zero_division(to_AjPhr_C, to_VePhr_C)),
            "ra_AjSuP_C": float(self.handle_zero_division(to_AjPhr_C, to_SuPhr_C)),
            "ra_AjPrP_C": float(self.handle_zero_division(to_AjPhr_C, to_PrPhr_C)),
            "ra_AjAvP_C": float(self.handle_zero_division(to_AjPhr_C, to_AvPhr_C)),
            "to_AvPhr_C": to_AvPhr_C,
            "as_AvPhr_C": float(self.handle_zero_division(to_AvPhr_C, n_sent)),
            "at_AvPhr_C": float(self.handle_zero_division(to_AvPhr_C, n_token)),
            "ra_AvNoP_C": float(self.handle_zero_division(to_AvPhr_C, to_NoPhr_C)),
            "ra_AvVeP_C": float(self.handle_zero_division(to_AvPhr_C, to_VePhr_C)),
            "ra_AvSuP_C": float(self.handle_zero_division(to_AvPhr_C, to_SuPhr_C)),
            "ra_AvPrP_C": float(self.handle_zero_division(to_AvPhr_C, to_PrPhr_C)),
            "ra_AvAjP_C": float(self.handle_zero_division(to_AvPhr_C, to_AjPhr_C)),
        }
        return result

    def extract_features(self, request, nlp):
        words, segments = self.process_input(request)
        # tokens = list(filter(None, tokens))

        context_start = words.index(self.tokenizer.eos_token)
        question_tokens = words[1:context_start]
        context_tokens = words[context_start + 2 : -1]
        question_ent_info = self.phrF(question_tokens, nlp)
        print(question_ent_info)
        # context_ent_info = self.entity_features(context_tokens, nlp)
        # tag_info = [(self.cls_token, 'SOS', 'SOS')] + \
        #            question_tag_info +\
        #            [(self.eos_token, 'EOS', 'EOS'),
        #             (self.eos_token, 'EOS', 'EOS')] + \
        #            context_tag_info + \
        #            [(self.eos_token, 'EOS', 'EOS')]
        #
        # assert len(tag_info) == len(words), \
        #     "tags and words not equal"
        # # print(tag_info)
        # instance_info = {'words': words, 'segments': segments, 'tags': tag_info}
        # # print(instance_info)
        # return instance_info


if __name__ == "__main__":
    model_path = "./src/checkpoints/roberta_squad"
    # model = RobertaModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    data = dataloader.PreprocessData(
        "squad_adversarial", "AddSent", save_data=False, save_path="../../../../"
    )
    outputs = list()
    tagger = FeatureExtractor(tokenizer=tokenizer)
    nlp = spacy.load("en_core_web_sm")

    c = 0
    processed_instances = OrderedDict()
    for ex in tqdm(data.processed_val_set()):
        # ex = remove_white_space(ex)
        try:
            # if ex["id"] == "56e1239acd28a01900c67641":
            #     print(ex)
            tag_info = tagger.extract_features(
                request={
                    "id": ex["id"],
                    "question": ex["question"],
                    "context": ex["context"],
                },
                nlp=nlp,
            )
            processed_instances[ex["id"]] = tag_info
            # print(tag_info)
            c += 1
        except Exception as e:
            print(f"Unable to get tags: {e}")
            print(ex)
        # if c == 1:
        break
    # utils.dump_to_bin(processed_instances,
    #                   "./src/results/squad_tag_info.bin")
    #
    # print(f"Saved instances: {c}")
