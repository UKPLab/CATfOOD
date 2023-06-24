import string
from typing import List, Tuple
from collections import OrderedDict

import torch
import spacy
from spacy.tokens import Doc
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    # RobertaAdapterModel,
)

from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from src.calibration.baseline import dataloader
import utils

BASE_PATH = "/storage/xyz/work/anon/research_projects/exp_calibration/"


class POSTagger:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.eos_token

    def encode(self,
               inputs: list = None,
               add_special_tokens: bool = True,
               return_tensors=None):
        """
        Encode inputs using the model tokenizer
        Args:
            inputs: question, context pair as a list
            add_special_tokens: where to add CLS, SEP tokens
            return_tensors: whether to return tensors
        Return:
            tokenized inputs
        """
        return self.tokenizer(inputs,
                              add_special_tokens=add_special_tokens,
                              return_tensors=return_tensors,
                              padding=True,
                              truncation=True,
                              max_length=512)

    def decode(self, input_ids: torch.Tensor, skip_special_tokens: bool) -> List[str]:
        """
        Decode received input_ids into a list of word tokens.
        Args:
            input_ids (torch.Tensor): Input ids representing
            word tokens for a sentence/document.
        """
        return self.tokenizer.convert_ids_to_tokens(
            input_ids[0],
            skip_special_tokens=skip_special_tokens
        )

    def _bpe_decode(
            self,
            tokens: List[str],
    ) -> Tuple[List[str], List]:

        byte_encoder = bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}
        decoded_each_tok = [
            bytearray([byte_decoder[c] for c in t]).decode(
                encoding="utf-8",
                errors="replace") for t in tokens
        ]

        end_points = []
        force_break = False
        for idx, token in enumerate(decoded_each_tok):
            # special token, punctuation, alphanumeric
            if token in self.tokenizer.all_special_tokens or \
                    token in string.punctuation or \
                    not any([x.isalnum() for x in token.lstrip()]) or \
                    token.lstrip == "'s":
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
            filtered_tokens.append(''.join(decoded_each_tok[s0:s1]))
        # print(filtered_tokens)
        # print(segments)

        return filtered_tokens, segments

    def process_input(self, request):
        inputs = [[request["question"], request["context"]]]
        encoded_inputs = self.encode(inputs,
                                     add_special_tokens=True,
                                     return_tensors="pt")

        decoded_text = self.decode(encoded_inputs["input_ids"],
                                   skip_special_tokens=False)
        # print(decoded_text)
        filtered_tokens, segments = self._bpe_decode(decoded_text)
        return filtered_tokens, segments

    def assign_pos_tags(self, tokens, nlp):
        words = [x.lstrip() for x in tokens]
        spaces = [False if i == len(tokens) - 1
                  else tokens[i + 1][0] == ' ' for i in range(len(tokens))]

        valid_idx = [i for i, w in enumerate(words) if len(w)]
        words = [words[i] for i in valid_idx]
        spaces = [spaces[i] for i in valid_idx]
        doc = Doc(nlp.vocab, words=words, spaces=spaces)
        processed_tokens = nlp(doc)

        tag_info = [('', 'NULL', 'NULL')] * len(tokens)
        for i, proc_tok in zip(valid_idx, processed_tokens):
            tag_info[i] = (proc_tok.text, proc_tok.pos_, proc_tok.tag_)

        return tag_info

    def tag_instance(self, request, nlp):
        words, segments = self.process_input(request)
        # tokens = list(filter(None, tokens))

        context_start = words.index(self.tokenizer.eos_token)
        question_tokens = words[1:context_start]
        context_tokens = words[context_start + 2: -1]
        question_tag_info = self.assign_pos_tags(question_tokens, nlp)
        context_tag_info = self.assign_pos_tags(context_tokens, nlp)
        tag_info = [(self.cls_token, 'SOS', 'SOS')] + \
                   question_tag_info +\
                   [(self.eos_token, 'EOS', 'EOS'),
                    (self.eos_token, 'EOS', 'EOS')] + \
                   context_tag_info + \
                   [(self.eos_token, 'EOS', 'EOS')]

        assert len(tag_info) == len(words), \
            "tags and words not equal"
        # print(tag_info)
        instance_info = {'words': words, 'segments': segments, 'tags': tag_info}
        # print(instance_info)
        return instance_info


if __name__ == "__main__":
    # model_path = "./src/checkpoints/roberta_squad"
    # model_path = "./roberta-squad"
    # model = RobertaModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    #
    # data = dataloader.PreprocessData("squad_adversarial",
    #                                  "AddSent",
    #                                  save_data=False,
    #                                  save_path="../../")
    # outputs = list()
    # tagger = POSTagger(tokenizer=tokenizer)
    # nlp = spacy.load("en_core_web_sm")
    # c = 0
    # processed_instances = OrderedDict()
    # # process OOD data
    # for ex in tqdm(data.processed_val_set()):
    #     # ex = remove_white_space(ex)
    #     try:
    #         # if ex["id"] == "56e1239acd28a01900c67641":
    #         #     print(ex)
    #         tag_info = tagger.tag_instance(
    #                         request={
    #                                     "id": ex["id"],
    #                                     "question": ex["question"],
    #                                     "context": ex["context"],
    #                                  },
    #                         nlp=nlp
    #                     )
    #         processed_instances[ex["id"]] = tag_info
    #         # print(tag_info)
    #         c += 1
    #     except Exception as e:
    #         print(f"Unable to get tags: {e}")
    #         print(ex)

    # print(f"Processed {c} instances of original data")
    # # process counterfactuals
    # cf_path = BASE_PATH + "src/data/squad_adversarial/rag_counterfactuals_turk0_last.jsonl"
    # with jsonlines.open(cf_path) as reader:
    #     for ex in tqdm(reader):
    #         try:
    #             tag_info = tagger.tag_instance(
    #                             request={
    #                                         "id": ex["id"].replace("_", "-cf-"),
    #                                         "question": ex["question"],
    #                                         "context": ex["context"],
    #                                      },
    #                             nlp=nlp
    #                         )
    #             processed_instances[ex["id"].replace("_", "-cf-")] = tag_info
    #             # print(tag_info)
    #             c += 1
    #         except Exception as e:
    #             print(f"Unable to get cf tags: {e}")
    #             print(ex)
    #
    #     print(f"Processed {c} instances of counterfactual data")
    #
    #     # if c == 1:
    #         # break
    # utils.dump_to_bin(processed_instances,
    #                   BASE_PATH+"src/data/squad_adv_new/pos_info.bin")
    #
    # print(f"Saved instances: {c}")

    ### Trivia QA
    def remove_white_space(example):
        example["question_text"] = ' '.join(example["question_text"].split())
        example["context_text"] = ' '.join(example["context_text"].split())
        return example

    data = dataloader.get_dev_examples(BASE_PATH + "src/data", "dev_trivia.json")
    # data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/NewsQA.jsonl")
    # data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/BioASQ-dev.jsonl")
    # data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/NaturalQuestionsShort.jsonl")

    outputs = list()
    tagger = POSTagger(tokenizer=tokenizer)
    nlp = spacy.load("en_core_web_sm")
    c = 0
    processed_instances = OrderedDict()
    # process OOD data
    for ex in tqdm(data):
        # ex = remove_white_space(ex)
        # print(ex)
        # break
        try:
            # if ex["qas_id"] == "6b67d06dbf4b4e2eae802b3e341ce8d4":
            #     print(ex)
            tag_info = tagger.tag_instance(
                request={
                    "id": ex["qas_id"],
                    "question": ex["question_text"],
                    "context": ex["context_text"],
                },
                nlp=nlp
            )
            processed_instances[ex["qas_id"]] = tag_info
            # print(tag_info)
            c += 1
        except Exception as e:
            print(f"Unable to get tags: {e}")
            print(ex)
    #
    # print(f"Processed {c} instances of original data")
    # c = 0
    # # process counterfactuals
    # cf_path = BASE_PATH + "src/data/natural_questions/rag_counterfactuals_turk0.jsonl"
    # with jsonlines.open(cf_path) as reader:
    #     for ex in tqdm(reader):
    #         try:
    #             tag_info = tagger.tag_instance(
    #                             request={
    #                                         "id": ex["id"].replace("_", "-cf-"),
    #                                         "question": ex["question"],
    #                                         "context": ex["context"],
    #                                      },
    #                             nlp=nlp
    #                         )
    #             processed_instances[ex["id"].replace("_", "-cf-")] = tag_info
    #             # print(tag_info)
    #             c += 1
    #         except Exception as e:
    #             print(f"Unable to get cf tags: {e}")
    #             print(ex)
    #
    #     print(f"Processed {c} instances of counterfactual data")
    #
    #     # if c == 1:
    #         # break
    utils.dump_to_bin(processed_instances,
                      BASE_PATH +"src/data/trivia_qa/pos_info_wo_space.bin")
    print(f"Saved instances: {c}")
