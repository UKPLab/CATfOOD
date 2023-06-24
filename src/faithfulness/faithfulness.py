# evaluation metrics for post-hoc explanation methods

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List
import ast
import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)
import torch
from collections import OrderedDict

from src.calibration.baseline import dataloader, utils

BASE_PATH = "/storage/ukp/work/anon/research_projects/exp_calibration/"

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


def comprehensiveness(
        full_text_pred,
        masked_text_pred
):
    """
    Model confidence score before and after removing top k% words

    Args:
        full_text_pred: pred score for the original text
        masked_text_pred: pred score for the masked text
    :return: Comprehensiveness score
    """
    comp_score = 0
    # higher score is better
    for score in masked_text_pred:
        # print(masked_text_pred)
        comp_score += np.round(np.maximum(0, full_text_pred - score), 4)
        # print(comp_score)

    avg_comp = comp_score/len(masked_text_pred)
    # print(avg_comp)
    return avg_comp


def sufficiency(
        full_text_pred,
        masked_text_pred
):
    """
        Model confidence score before and after keeping top k% words

        Args:
            full_text_pred: pred score for the original text
            masked_text_pred: pred score for the masked text
        :return: Sufficiency score
        """
    sufficiency_score = 1 - comprehensiveness(full_text_pred, masked_text_pred)
    return sufficiency_score


def log_odds():
    pass

class FaithfulEval:

    def __init__(self, dataset, method, model_type, model, tokenizer):
        self.dataset = dataset
        self.method = method
        self.model_type = model_type
        self.model = AutoModelForQuestionAnswering.from_pretrained(BASE_PATH+model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.path = f"src/data/{self.dataset}"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.load_data()

        if self.dataset == "squad_adversarial":
            loader = dataloader.PreprocessData("squad_adversarial", "AddSent", save_data=False, save_path="../../../")
            data = loader.processed_val_set()
        elif self.dataset == "trivia_qa":
            data = dataloader.get_dev_examples("./src/data", "dev_trivia.json")
        elif self.dataset == "hotpot_qa":
            data = dataloader.get_dev_examples("./src/data", "dev_hotpot.json")
        elif self.dataset == "news_qa":
            data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/NewsQA.jsonl")
        elif self.dataset == "bioasq":
            data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/BioASQ-dev.jsonl")
        elif self.dataset == "natural_questions":
            data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/NaturalQuestionsShort.jsonl")
        else:
            raise ValueError("Dataset not supported.")

        self.data = data

    def remove_white_space(self, example):
        example["question_text"] = ' '.join(example["question_text"].split())
        example["context_text"] = ' '.join(example["context_text"].split())
        return example

    def load_interp_info(self, file_dict, qas_id):
        return torch.load(file_dict[qas_id])

    def build_file_dict(self, dataset, model_type, method):
        # hard-coded path here: be careful
        # prefix = 'squad_sample-addsent_roberta-base'
        prefix = f'{dataset}/dev/roberta'
        fnames = os.listdir(os.path.join(f'exp_roberta_{model_type}', method, prefix))
        # print(fnames)
        qa_ids = [x.split('.')[0] for x in fnames]
        # exp_roberta_flan_ul2_context_noise_rel
        fullnames = [os.path.join(f'exp_roberta_{model_type}', method, prefix, x) for x in fnames]
        return dict(zip(qa_ids, fullnames))

    def load_data(self):
        self.tagger_info = utils.load_bin(f"{self.path}/pos_info.bin")
        if self.model_type.__contains__("llama") or self.model_type.__contains__("gpt_neox"):
            self.pred_df = pd.read_csv(f"{self.path}/outputs_{self.model_type}_filter.csv")
        elif self.model_type.__contains__("flan_ul2"):
            self.pred_df = pd.read_csv(f"{self.path}/outputs_flan_ul2_context_rel_noise_filter.csv")
        elif self.model_type.__contains__("base"):
            self.pred_df = pd.read_csv(f"{self.path}/outputs_wo_cf_roberta.csv")
        elif self.model_type.__contains__("rag"):
            self.pred_df = pd.read_csv(f"{self.path}/outputs_rag.csv")
        self.predictions = self.pred_df.set_index('id').T.to_dict('dict')

        if self.method == "shap":
            self.attributions = self.build_file_dict(
                dataset=self.dataset, model_type=self.model_type.split("_")[0], method=self.method
            )
        else:
            self.attributions = utils.load_bin(f"{self.path}/{self.method}_info_{self.model_type}.bin")

    def encode(self, inputs: list = None,
               add_special_tokens: bool = True,
               return_tensors=None):
        """
        Encode inputs using the model tokenizer
        Args:
            inputs: question, context pair as a list
            add_special_tokens: where to add CLS, SEP tokens
            return_offsets_mapping: mapping of sub-words to tokens
            return_tensors: whether to return tensors
        Return:
            tokenized inputs
        """
        return self.tokenizer(inputs,
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
            input_ids[0],
            skip_special_tokens=skip_special_tokens
        )

    def _ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be :obj:`torch.Tensor`): The
                tensors to place on :obj:`self.device`.

        Return:
            :obj:`Dict[str, torch.Tensor]`: The same as :obj:`inputs`
                but on the proper device.
        """
        return {name: tensor.to(self.model.device) for name, tensor in inputs.items()}


    def _predict(
            self,
            inputs,
            **model_kwargs
    ) -> tuple:
        """
        Inference on the input.
        Args:
            inputs: list of question, context input
        Returns:
             The model outputs and optionally the input features
        """
        # encoded_inputs = self.encode(inputs,
        #                              add_special_tokens=True,
        #                              return_tensors="pt")
        # encoded_inputs.to(self.device)
        # self.decoded_text = self.decode(
        #     encoded_inputs["input_ids"],
        #     skip_special_tokens=False
        # )
        # self.words_mapping = encoded_inputs.word_ids()

        all_predictions = list()
        self.model.to(self.device)
        predictions = self.model(
            **inputs,
            **model_kwargs
        )
        # print(predictions)
        all_predictions.append(predictions)
        keys = all_predictions[0].keys()
        # print(all_predictions)
        final_prediction = {}
        for key in keys:
            if isinstance(all_predictions[0][key], tuple):
                tuple_of_lists = list(
                    zip(*[[torch.stack(p).to(self.device)
                           if isinstance(p, tuple) else p.to(self.device)
                           for p in tpl[key]] for tpl in all_predictions]))
                final_prediction[key] = tuple(torch.cat(l) for l in tuple_of_lists)
            else:
                final_prediction[key] = torch.cat(
                    [
                        p[key].to(self.device)
                        for p in all_predictions
                    ])

        return predictions, inputs

    def question_answering(self, request, top_k):
        """
        Span-based question answering for a given question and context.
        We expect the input to use the (question, context) format for the text pairs.
        Args:
          request: the prediction request

        """
        def decode(start_: np.ndarray,
                   end_: np.ndarray,
                   top_k: int,
                   max_answer_len: int,
                   undesired_tokens_: np.ndarray) -> Tuple:

            """
            Take the output of any :obj:`ModelForQuestionAnswering` and will generate probabilities
            for each span to be the actual answer.

            In addition, it filters out some unwanted/impossible cases like answer len being greater
            than max_answer_len or answer end position being before the starting position. The method
            supports output the k-best answer through the topk argument.

            Args:
                start_ (:obj:`np.ndarray`): Individual start probabilities for each token.
                end (:obj:`np.ndarray`): Individual end_ probabilities for each token.
                topk (:obj:`int`): Indicates how many possible answer span(s) to extract from the model output.
                max_answer_len (:obj:`int`): Maximum size of the answer to extract from the model's output.
                undesired_tokens_ (:obj:`np.ndarray`): Mask determining tokens that can be part of the answer
            """
            # Ensure we have batch axis
            if start_.ndim == 1:
                start_ = start_[None]

            if end_.ndim == 1:
                end_ = end_[None]

            # Compute the score of each tuple(start_, end_) to be the real answer
            outer = np.matmul(np.expand_dims(start_, -1), np.expand_dims(end_, 1))

            # Remove candidate with end_ < start_ and end_ - start_ > max_answer_len
            candidates = np.tril(np.triu(outer), max_answer_len - 1)

            #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
            # scores_flat = candidates.flatten()
            # if top_k == 1:
            #     idx_sort = [np.argmax(scores_flat)]
            # elif len(scores_flat) < top_k:
            #     idx_sort = np.argsort(-scores_flat)
            # else:
            #     idx = np.argpartition(-scores_flat, top_k)[0:top_k]
            #     idx_sort = idx[np.argsort(-scores_flat[idx])]

            # starts_, ends_ = np.unravel_index(idx_sort, candidates.shape)[1:]
            # print(starts_, ends_)
            starts_, ends_ = np.array(answer_pos[0]), np.array(answer_pos[1])
            desired_spans = np.isin(starts_, undesired_tokens_.nonzero()) \
                            & np.isin(ends_, undesired_tokens_.nonzero())
            starts_ = starts_[desired_spans]
            ends_ = ends_[desired_spans]
            scores_ = candidates[0, starts_, ends_]


            replacement = np.array([])  # Replacement array
            if np.array_equal(starts_, [0]) and  np.array_equal(ends_, [0]):
                starts_ = replacement
                ends_ = replacement
            if np.array_equal(scores_, [0]):
                scores_ = replacement
            #
            # print(starts_, ends_)
            # print(scores_)

            return starts_, ends_, scores_

        # print(request)
        predictions, features = self._predict(request["inputs"])
        # print(predictions)

        task_outputs = {"answers": []}
        for idx, (start, end, context, answer_pos) in enumerate(zip(predictions["start_logits"],
                                                             predictions["end_logits"],
                                                             request["context"], request["answer_pos"])):
            start = start.cpu().detach().numpy()
            end = end.cpu().detach().numpy()
            # print(features.sequence_ids(idx))
            # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
            question_tokens = np.abs(np.array([s != 1 for s in features.sequence_ids(idx)]) - 1)
            # Unmask CLS token for 'no answer'
            question_tokens[0] = 1
            undesired_tokens = question_tokens & features["attention_mask"][idx].detach().cpu().numpy()

            # Generate mask
            undesired_tokens_mask = undesired_tokens == 0.0

            # Make sure non-context indexes in the tensor cannot contribute to the softmax
            start = np.where(undesired_tokens_mask, -10000.0, start)
            end = np.where(undesired_tokens_mask, -10000.0, end)

            start = np.exp(start - np.log(np.sum(np.exp(start), axis=-1, keepdims=True)))
            end = np.exp(end - np.log(np.sum(np.exp(end), axis=-1, keepdims=True)))

            # print(start, end)
            # print(answer_pos)


            # Get score for 'no answer' then mask for decoding step (CLS token
            no_answer_score = (start[0] * end[0]).item()
            start[0] = end[0] = 0.0

            starts, ends, scores = decode(
                start, end, top_k, 128, undesired_tokens
            )

            enc = features[idx]
            # print(enc)
            answers = [
                {
                    "score": score.item(),
                    "start": enc.word_to_chars(
                        enc.token_to_word(s), sequence_index=1)[0],
                    "end": enc.word_to_chars(enc.token_to_word(e), sequence_index=1)[1],
                    "answer": context[
                              enc.word_to_chars(enc.token_to_word(s), sequence_index=1)[0]:
                              enc.word_to_chars(enc.token_to_word(e), sequence_index=1)[1]],
                }
                for s, e, score in zip(starts, ends, scores)]
            # answers.append({"score": no_answer_score, "start": 0, "end": 0, "answer": ""})
            answers = sorted(answers, key=lambda x: x["score"], reverse=True)  #[:topk]
            # print(answers)
            task_outputs["answers"].append(answers)
        return task_outputs

    def alter_attention_mask(self, attention_mask, zero_indices):
        # print(attention_mask)
        mask = attention_mask
        for i in range(len(mask[0])):
            # set the indices for the answer tokens or tokens that are not in the zero_list to 1
            # if (answer_position_list[0] <= i <= answer_position_list[-1]) or (i not in zero_indices):
            #     mask[0][i] = 1
            # set the indices for the tokens to mask to 0
            if i in zero_indices:
                mask[0][i] = 0
        attention_mask = torch.tensor(data=mask, device=self.device)
        # print(attention_mask)
        return attention_mask

    def evaluate(self, topk, metric, context_only=True):

        masked_pred_score = []
        processed_instances = OrderedDict()
        c = 0
        for ex in tqdm(self.data):
            if self.dataset not in ["squad", "squad_adversarial"]:
                ex = self.remove_white_space(ex)
                # pass

            if self.dataset not in ["squad", "squad_adversarial"]:
                q_id = ex["qas_id"]
                question = ex["question_text"]
                context = ex["context_text"]
                answer = ex["answer_text"]
                # pred = self.predictions[q_id]
            else:
                q_id = ex["id"]
                question = ex["question"]
                context = ex["context"]
                answer = ex["answers"]["text"]
            # print(answer)
            pred = self.predictions[q_id]
            predictions = ast.literal_eval(pred["pred_text"])

            attributions = self.attributions[q_id]
            # print(attributions)
            importance = attributions["attributions"]

            pred_score = predictions[0][0]["score"]
            pred_text = predictions[0][0]["answer"]
            # print("Original:", pred_text)
            answer_start, answer_end = \
                ast.literal_eval(pred["answer_start"])[0], ast.literal_eval(pred["answer_end"])[0]
            inputs = self.encode([[question, context]])

            if context_only:
                context_start = inputs["input_ids"][0].index(self.tokenizer.sep_token_id)
                filtered_attributions = importance[context_start + 2:]
            else:
                filtered_attributions = importance

            if metric == "suff":
                topk = topk[::-1]

            for k in topk:
                # print(k)
                # batch_inputs = {"input_ids": [], "attention_mask": []}
                if metric == "comp":
                    num_tokens_to_mask = int(len(importance)*(k/100))
                else:
                    # for suff, keep topk tokens only, so we'll mask (100-topk) % tokens
                    num_tokens_to_mask = int(len(importance) * ((100-k) / 100))

                orig_mask = inputs["attention_mask"]
                # print("attn mask", orig_mask)
                assert len(importance) == len(orig_mask[0])
                rankings = np.argsort(filtered_attributions)  # ascending sort
                # print(rankings)

                if metric == "comp":
                    # remove answer indices from rankings
                    attention_mask = orig_mask
                    answer_position_list = range(answer_start, answer_end)
                    filtered_list = [x for x in rankings if x not in answer_position_list]
                    # print("fil 1", filtered_list)
                    filtered_list = filtered_list[::-1]
                    # print(filtered_list)
                    zero_indices = filtered_list[:num_tokens_to_mask]
                    altered_attn_mask = self.alter_attention_mask(attention_mask, zero_indices)
                    # print(altered_attn_mask)
                else:
                    # filtered_list = rankings
                    attention_mask = orig_mask
                    answer_position_list = range(answer_start, answer_end)
                    filtered_list = [x for x in rankings if x not in answer_position_list]
                    # print("fil 1", filtered_list)
                    # print("num", num_tokens_to_mask)
                    # filtered_list = filtered_list[::]
                    # print(filtered_list)
                    zero_indices = filtered_list[:num_tokens_to_mask]
                    # print(zero_indices)
                    altered_attn_mask = self.alter_attention_mask(attention_mask, zero_indices)
                    # print("alter mask:", altered_attn_mask)

                inputs["input_ids"] = torch.tensor(inputs["input_ids"]).to(self.device)
                inputs["attention_mask"] = altered_attn_mask

                request = dict()
                request["inputs"] = inputs
                # reset original mask
                # inputs["attention_mask"] = orig_mask
                request["answer_pos"] = [[answer_start, answer_end]]
                request["context"] = [context]
                # print(inputs)
                # # print(request)
                outputs = self.question_answering(request, top_k=1)
                # print("outputs:", outputs)
                answers = outputs["answers"][0]
                if answers:
                    ans_score = answers[0]["score"]
                else:
                    ans_score = 0.0
                # ans_score = [ans["score"] if normalize_answer(pred_text) == normalize_answer(ans["answer"]) else 0.0 for
                #              ans in outputs["answers"][0]]
                # print("Ans score:", ans_score)
                # first_non_zero = next((element for element in ans_score if element != 0), 0.0)
                masked_pred_score.append(ans_score)
            # print("-"*8)

            if metric == "comp":
                score = comprehensiveness(pred_score, masked_pred_score)
            else:
                score = sufficiency(pred_score, masked_pred_score)
                # print(score)
            processed_instances[q_id] = score
            # print(processed_instances)
            # c+=1
            # if c==10:
            #     break
        utils.dump_to_bin(processed_instances,
                          BASE_PATH + f"src/data/{self.dataset}/{self.method}_{metric}_{self.model_type}_5.bin")

    def get_mean_score(self, metric):
        sum_score, mean_score = 0, 0
        scores = utils.load_bin(BASE_PATH + f"src/data/{self.dataset}/{self.method}_{metric}_{self.model_type}_5.bin")
        for ex in tqdm(self.data):
            if self.dataset not in ["squad", "squad_adversarial"]:
                ex = self.remove_white_space(ex)

            if self.dataset not in ["squad", "squad_adversarial"]:
                q_id = ex["qas_id"]
            else:
                q_id = ex["id"]
            sum_score+=scores[q_id]

        mean_score = sum_score/len(scores)
        return round(mean_score, 2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Passing arguments for model, tokenizer, and dataset.")

    parser.add_argument(
        "--model_name",
        default="",
        type=str, required=True, help="Specify the model to use.")
    parser.add_argument("--tokenizer", default="roberta-base", type=str, required=False,
                        help="Specify the tokenizer to use.")
    parser.add_argument("--dataset", type=str, required=True, help="Specify the dataset to use.")
    parser.add_argument("--model_type", type=str, required=True, help="Specify the model type to use.")
    parser.add_argument("--metric", type=str, required=True, help="Specify the evaluation metric to use.")
    parser.add_argument("--get_score", action="store_true", help="Whether to get mean score.")

    args = parser.parse_args()

    methods = ["attn", "sc_attn", "simple_grads", "ig"]
    # methods = ["attn"]

    for method in methods:
        eval = FaithfulEval(
            dataset=args.dataset,
            method=method,
            model_type=args.model_type,
            model=args.model_name,
            tokenizer=args.tokenizer,
        )
        topk = [2, 10, 20, 50]
        # topk = [10]
        if args.get_score:
            score = eval.get_mean_score(metric=args.metric)
            print("Method: ", method)
            print("Score: ", score)
        else:
            eval.evaluate(topk=topk, metric=args.metric, context_only=True)
