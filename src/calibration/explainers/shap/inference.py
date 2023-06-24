import numpy as np
import os
import json
import collections
from tqdm import tqdm
from typing import List, Tuple

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoTokenizer,
    RobertaForQuestionAnswering,
    # BertAdapterModel,
    # RobertaAdapterModel
)
import dataloader

# BASE_PATH = "/storage/xyz/work/anon/research_projects/exp_calibration/"
BASE_PATH = "//"

class Inference:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

        if self.model.config.model_type == "gpt2":
            self.ref_token_id = self.tokenizer.eos_token_id
        else:
            self.ref_token_id = self.tokenizer.pad_token_id

        self.sep_token_id = (
            self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.eos_token_id
        )
        self.cls_token_id = (
            self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.bos_token_id
        )

        self.model_prefix = model.base_model_prefix

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be :obj:`torch.Tensor`): The tensors to place on :obj:`self.device`.

        Return:
            :obj:`Dict[str, torch.Tensor]`: The same as :obj:`inputs` but on the proper device.
        """
        return {name: tensor.to(self.model.device) for name, tensor in inputs.items()}

    def encode(self, inputs: list = None, add_special_tokens: bool = True, return_tensors=None):
        """
        Encode inputs using the model tokenizer
        Args:
            inputs: question, context pair as a list
            add_special_tokens: where to add CLS, SEP tokens
            return_tensors: whether to return tensors
        Return:
            tokenized inputs
        """
        return self.tokenizer(inputs, add_special_tokens=add_special_tokens, return_tensors=return_tensors,
                              padding=True, truncation=True, max_length=512)

    def decode(self, input_ids: torch.Tensor, skip_special_tokens: bool) -> List[str]:
        """
        Decode received input_ids into a list of word tokens.
        Args:
            input_ids (torch.Tensor): Input ids representing
            word tokens for a sentence/document.
        """
        return self.tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=skip_special_tokens)

    def _predict(
            self,
            request,
            batch_size: int = 1
            ) -> tuple:
        """
        Inference on the input.
        Args:
            inputs: list of question, context input
        Returns:
             The model outputs and optionally the input features
        """
        all_predictions = []
        # self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        features = self.encode(request,
                               add_special_tokens=True,
                               return_tensors="pt",
                            )
        print("tokens", features.tokens())
        print("input_ids", features)

        for start_idx in range(0, len(request), batch_size):
            with torch.no_grad():
                input_features = {k: features[k][start_idx:start_idx + batch_size] for k in features.keys()}
                input_features = self._ensure_tensor_on_device(**input_features)
                predictions = self.model(**input_features)
                all_predictions.append(predictions)
        keys = all_predictions[0].keys()
        final_prediction = {}
        for key in keys:
            if isinstance(all_predictions[0][key], tuple):
                tuple_of_lists = list(
                    zip(*[[torch.stack(p).to(self.device) if isinstance(p, tuple) else p.to(self.device)
                           for p in tpl[key]] for tpl in all_predictions]))
                final_prediction[key] = tuple(torch.cat(l) for l in tuple_of_lists)
            else:
                final_prediction[key] = torch.cat([p[key].to(self.device) for p in all_predictions])
        # print(final_prediction)
        # print(features)
        return final_prediction, features

    def question_answering(self, request, top_k: int = 1):
        """
        Span-based question answering for a given question and context.
        We expect the input to use the (question, context) format for the text pairs.
        Args:
          request: the prediction request
          top_k: num preds

        """

        def decode(start_: np.ndarray,
                   end_: np.ndarray,
                   topk: int,
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
            scores_flat = candidates.flatten()
            if topk == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < topk:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, topk)[0:topk]
                idx_sort = idx[np.argsort(-scores_flat[idx])]

            starts_, ends_ = np.unravel_index(idx_sort, candidates.shape)[1:]
            desired_spans = np.isin(starts_, undesired_tokens_.nonzero()) & np.isin(ends_, undesired_tokens_.nonzero())
            starts_ = starts_[desired_spans]
            ends_ = ends_[desired_spans]
            scores_ = candidates[0, starts_, ends_]

            return starts_, ends_, scores_

        predictions, features = self._predict(request)
        start_idx = torch.argmax(predictions["start_logits"])
        end_idx = torch.argmax(predictions["end_logits"])
        answer_start = torch.tensor([start_idx])
        answer_end = torch.tensor([end_idx])

        # print(features)
        task_outputs = {"answers": []}
        for idx, (start, end, (_, context)) in enumerate(zip(predictions["start_logits"],
                                                             predictions["end_logits"],
                                                             request)):
            start = start.cpu().detach().numpy()
            end = end.cpu().detach().numpy()
            # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
            question_tokens = np.abs(np.array([s != 1 for s in features.sequence_ids(idx)]) - 1)
            # Unmask CLS token for 'no answer'
            question_tokens[0] = 1
            undesired_tokens = question_tokens & features["attention_mask"][idx].numpy()

            # Generate mask
            undesired_tokens_mask = undesired_tokens == 0.0

            # Make sure non-context indexes in the tensor cannot contribute to the softmax
            start = np.where(undesired_tokens_mask, -10000.0, start)
            end = np.where(undesired_tokens_mask, -10000.0, end)

            start = np.exp(start - np.log(np.sum(np.exp(start), axis=-1, keepdims=True)))
            end = np.exp(end - np.log(np.sum(np.exp(end), axis=-1, keepdims=True)))

            # Get score for 'no answer' then mask for decoding step (CLS token
            no_answer_score = (start[0] * end[0]).item()
            start[0] = end[0] = 0.0

            starts, ends, scores = decode(
                start, end, top_k, 128, undesired_tokens
            )
            enc = features[idx]
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

            answers.append({"score": no_answer_score, "start": 0, "end": 0, "answer": ""})
            answers = sorted(answers, key=lambda x: x["score"], reverse=True)[:top_k]
            task_outputs["answers"].append(answers)
            task_outputs["answer_start"] = answer_start.cpu().detach().numpy()
            task_outputs["answer_end"] = answer_end.cpu().detach().numpy()
        return task_outputs


if __name__ == '__main__':
    # base_model = "roberta-base"
    # adapter_model = "AdapterHub/roberta-base-pf-squad"
    # model = RobertaAdapterModel.from_pretrained(base_model)
    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    # adapter_name = model.load_adapter(adapter_model, source="hf")
    # model.active_adapters = adapter_name

    model_path = BASE_PATH+"src/checkpoints/roberta-squad-with-easy-0.75"
    # model_path = BASE_PATH+"roberta-squad-with-cfs-15k"
    model = RobertaForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    inf = Inference(model=model, tokenizer=tokenizer)

    # data = dataloader.PreprocessData("squad_adversarial", "AddSent", save_data=False, save_path="../../")

    # batch_data = [for ex in data.processed_val_set()]
    # def group_batch(batch):
    #     return {k: [v] for k, v in batch.items()}
    # data = data.processed_val_set().map(group_batch, batched=True, batch_size=2)
    # c=0
    # for ex in data:
    #     print(ex)
    #     c+=1
    #     if c==2:
    #         break
        # print(inf.predict(ex))

    # print(data)


    data = dataloader.get_dev_examples("./src/data", "dev_trivia.json")
    outputs = list()

    def remove_white_space(example):
        example["question_text"] = ' '.join(example["question_text"].split())
        example["context_text"] = ' '.join(example["context_text"].split())
        return example

    all_nbest_json = collections.OrderedDict()
    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "score", "answer_start", "answer_end"]
    )

    seen_predictions = {}
    nbest = []

    c = 0
    for ex in tqdm(data):
        # for ex in tqdm(data.processed_val_set()):
        # print(ex)
        ex = remove_white_space(ex)
        try:
            prediction = inf.question_answering(
                request=[[
                    ex["question_text"],
                    ex["context_text"],
                ]],
                top_k=20
            )
            # break
            # print(prediction)
            for i, pred in enumerate(prediction["answers"][0]):
                # print(pred)
                nbest.append(_NbestPrediction(
                    text=pred["answer"],
                    score=pred["score"],
                    answer_start=pred["start"],
                    answer_end=pred["end"]
                ))

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = entry.score
                output["answer_start"] = entry.answer_start
                output["answer_end"] = entry.answer_end
                nbest_json.append(output)

            nbest = []
            all_nbest_json[ex["id"]] = nbest_json
            # print(all_nbest_json)
            c += 1
        except Exception as e:
            print(f"Unable to get prediction: {e}")
    #     # if c == 10:
    #     #     break
    #
    path = f"{BASE_PATH}src/data/trivia_qa/"
    output_nbest_file = os.path.join(path, "nbest_predictions_roberta_easy_cfs_0.75.json")
    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    # print(outputs)
    #
    # cf_path = BASE_PATH + "src/data/squad_adversarial/rag_counterfactuals_turk0_last.jsonl"
    # with jsonlines.open(cf_path) as reader:
    #     for ex in tqdm(reader):
    #         try:
    #             prediction = inf.question_answering(
    #                 request=[[
    #                     ex["question"],
    #                     ex["context"],
    #                 ]],
    #                 top_k=20
    #             )
    #             result = {
    #                 "id": ex["id"].replace("_", "-cf-"),
    #                 "question": ex["question"],
    #                 "context": ex["context"],
    #                 "gold_text": [ex["answer"]],
    #                 "pred_text": prediction["answers"],
    #                 "answer_start": prediction["answer_start"],
    #                 "answer_end": prediction["answer_end"]
    #             }
    #             outputs.append(result)
    #             c+=1
    #         except Exception as e:
    #             print(f"Unable to get cf prediction: {e}")
    #             print(ex)
    #
    #     print(f"Processed {c} instances of counterfactual data")

    # df = pd.DataFrame(outputs)
    # df.to_csv(f"{BASE_PATH}src/data/squad_adv_new/outputs.csv")

    # Trivia QA
    # data = dataloader.get_dev_examples(BASE_PATH+"src/data", "dev_hotpot.json")
    # data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/NewsQA.jsonl")
    # data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/BioASQ-dev.jsonl")
    # data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/NaturalQuestionsShort.jsonl")
    # def remove_white_space(example):
    #     example["question_text"] = ' '.join(example["question_text"].split())
    #     example["context_text"] = ' '.join(example["context_text"].split())
    #     return example
    #
    # outputs = list()
    # c = 0
    # for ex in tqdm(data):
    #     # print(ex)
    #     ex = remove_white_space(ex)
    #     try:
    #         prediction = inf.question_answering(
    #             request=[[
    #                 ex["question_text"],
    #                 ex["context_text"],
    #             ]],
    #             top_k=20
    #         )
    #         result = {
    #             "id": ex["qas_id"],
    #             "question": ex["question_text"],
    #             "context": ex["context_text"],
    #             "gold_text": ex["answer_text"],
    #             "pred_text": prediction["answers"],
    #             "answer_start": prediction["answer_start"],
    #             "answer_end": prediction["answer_end"]
    #         }
    #         outputs.append(result)
    #         c += 1
    #     except Exception as e:
    #         print(f"Unable to get prediction: {e}")
    #
    # cf_path = BASE_PATH + "src/data/natural_questions/rag_counterfactuals_turk0.jsonl"
    # with jsonlines.open(cf_path) as reader:
    #     for ex in tqdm(reader):
    #         try:
    #             prediction = inf.question_answering(
    #                 request=[[
    #                     ex["question"],
    #                     ex["context"],
    #                 ]],
    #                 top_k=20
    #             )
    #             result = {
    #                 "id": ex["id"].replace("_", "-cf-"),
    #                 "question": ex["question"],
    #                 "context": ex["context"],
    #                 "gold_text": [ex["answer"]],
    #                 "pred_text": prediction["answers"],
    #                 "answer_start": prediction["answer_start"],
    #                 "answer_end": prediction["answer_end"]
    #             }
    #             outputs.append(result)
    #             c += 1
    #         except Exception as e:
    #             print(f"Unable to get cf prediction: {e}")
    #             print(ex)
    #
    #     print(f"Processed {c} instances of counterfactual data")
    #
    # df = pd.DataFrame(outputs)
    # df.to_csv(f"{BASE_PATH}src/data/natural_questions/outputs_roberta.csv")
