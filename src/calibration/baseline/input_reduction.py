# class to perturb input until the label changes

import numpy as np
import torch
from typing import Tuple
from transformers import (
    BertAdapterModel,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    BatchEncoding
)

from src.calibration.explainers import random_attribution
from src import shap_attributions
from src.calibration.explainers.gradients import simple_grads, integrated_grads
from src.calibration.explainers.attentions import attention, scaled_attention

torch.manual_seed(4)
torch.cuda.manual_seed(4)
np.random.seed(4)


class InputReduction:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 method: str,
                 request: dict):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.tokenizer = tokenizer
        self.question = request["question"]
        self.context = request["context"]
        self.top_k = request["top_k"]
        self.mode = request["mode"]
        self.method = method

        self.sep_token = (
            self.tokenizer.sep_token if self.tokenizer.sep_token is not None else self.tokenizer.eos_token
        )
        self.cls_token = (
            self.tokenizer.cls_token if self.tokenizer.cls_token is not None else self.tokenizer.bos_token
        )

    def compute_attributions(self):
        if self.method == "random":
            imp_scores = random_attribution.RandomAttributions(self.model, self.tokenizer)
            attributions = imp_scores.interpret([[self.question, self.context]], self.top_k, output="raw")
        elif self.method == "attention":
            grads = attention.AttnAttribution(self.model, self.tokenizer)
            attributions = grads.interpret([[self.question, self.context]], self.top_k, output="raw")
        elif self.method == "scaled_attn":
            grads = scaled_attention.ScaledAttention(self.model, self.tokenizer)
            attributions = grads.interpret([[self.question, self.context]], self.top_k, output="raw")
        elif self.method == "simple_grads":
            grads = simple_grads.SimpleGradients(self.model, self.tokenizer)
            attributions = grads.interpret([[self.question, self.context]], self.top_k, output="raw")
        elif self.method == "integrated_grads":
            grads = integrated_grads.IntegratedGradients(self.model, self.tokenizer)
            attributions = grads.interpret([[self.question, self.context]], self.top_k, output="raw")
        # elif self.method == "smooth_grads":
        #     grads = smooth_grads.SmoothGradients(self.model, self.tokenizer)
        #     attributions = grads.interpret([[self.question, self.context]], self.top_k, output="raw")
        elif self.method == "shap":
            grads = shap_attributions.SHAPAttributions(
                model=self.model,
                tokenizer=self.tokenizer,
                request=[[self.question, self.context]],
                visualize=False,
                topk=self.top_k
            )
            attributions = grads.run(output="raw")
        else:
            raise Exception("Attribution method not allowed")
        # print(attributions)
        return attributions

    def get_predictions(self):
        enc_inputs, attributions, answer_start, answer_end = self.compute_attributions()
        enc_inputs.to("cpu")
        # print(len(enc_inputs["input_ids"][0]), len(attributions))
        token_ids = enc_inputs["input_ids"]
        sep_idx = token_ids.tolist()[0].index(self.tokenizer.sep_token_id)
        # print(sep_idx)
        # don't mask sep and pad tokens
        mask = (token_ids != self.tokenizer.sep_token_id).long() & (token_ids != self.tokenizer.pad_token_id).long()
        instance_attribution = torch.tensor(attributions, dtype=torch.float32).to(self.device) + \
                               (1 - mask.to(self.device)) * 1e-10
        # get top_k % words
        num_words_to_mask = int((self.top_k/100) * instance_attribution.size()[1])
        if self.mode == "max":
            # get top_k attributions from context; can be changed later to include question too
            topk_idx = torch.topk(instance_attribution[0][sep_idx:], num_words_to_mask).indices
        elif self.mode == "min":
            topk_idx = torch.topk(instance_attribution[0][sep_idx:], num_words_to_mask, largest=False).indices
        # mask top_k tokens from context
        words_to_mask = torch.zeros(instance_attribution.size())
        words_to_mask[0][topk_idx + sep_idx] = 1
        tmp_mask = words_to_mask
        sum_before_mask = torch.sum(tmp_mask[0])
        # unmask answer and sep tokens
        words_to_mask[0][answer_start: answer_end+1] = 0
        sum_after_mask = torch.sum(words_to_mask[0])
        mask_diff = (sum_before_mask-sum_after_mask).cpu().detach().numpy()
        # if answer tokens are masked, we need to unmask them and mask other tokens
        if mask_diff > 0:
            topk_idx = torch.topk(instance_attribution[0][sep_idx:], num_words_to_mask+int(mask_diff)).indices
            # mask top_k tokens from context
            words_to_mask = torch.zeros(instance_attribution.size())
            words_to_mask[0][topk_idx + sep_idx] = 1
            # unmask answer and sep tokens
            words_to_mask[0][answer_start: answer_end + 1] = 0

        words_to_mask[0][sep_idx] = 0
        # words_to_mask.to(self.device)
        # mask features
        # print(token_ids, words_to_mask)
        inputs_masked = torch.tensor(token_ids * (1 - words_to_mask) + words_to_mask * self.tokenizer.mask_token_id,
                                     dtype=torch.int).to(self.device)
        # print(inputs_masked)
        dec_inputs = self.tokenizer.decode(torch.tensor(inputs_masked[0], dtype=torch.int))
        # print(dec_inputs)
        inputs = dec_inputs.split()
        filter_idx = list(inputs).index(self.sep_token)
        inputs = [value if self.sep_token not in value else value.replace(self.sep_token, "") for value in inputs]
        masked_context = " ".join(inputs[filter_idx:])
        # print(masked_context)
        output_new = self.question_answering([[self.question, masked_context]])
        outputs_old = self.question_answering([[self.question, self.context]])

        return outputs_old, output_new

    def _ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be :obj:`torch.Tensor`): The tensors to place on :obj:`self.device`.

        Return:
            :obj:`Dict[str, torch.Tensor]`: The same as :obj:`inputs` but on the proper device.
        """
        return {name: tensor.to(self.model.device) for name, tensor in inputs.items()}

    def _predict(self, request, batch_size=1) \
            -> Tuple[dict, BatchEncoding]:
        """
        Inference on the input.

        Args:
         request: the request with the input and optional kwargs
         batch_size: input batch size

        Returns:
             The model outputs and optionally the input features
        """
        all_predictions = []
        # self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        features = self.tokenizer(request,
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=True,
                                  max_length=512)
        for start_idx in range(0, len(request), batch_size):
            with torch.no_grad():
                input_features = {k: features[k][start_idx:start_idx+batch_size] for k in features.keys()}
                input_features = self._ensure_tensor_on_device(**input_features)
                predictions = self.model(**input_features)
                all_predictions.append(predictions)
        keys = all_predictions[0].keys()
        final_prediction = {}
        for key in keys:
            if isinstance(all_predictions[0][key], tuple):
                tuple_of_lists = list(zip(*[[torch.stack(p).to(self.device) if isinstance(p, tuple) else p.to(self.device)
                                             for p in tpl[key]] for tpl in all_predictions]))
                final_prediction[key] = tuple(torch.cat(l) for l in tuple_of_lists)
            else:
                final_prediction[key] = torch.cat([p[key].to(self.device) for p in all_predictions])
        # print(final_prediction)
        # print(features)
        return final_prediction, features

    def question_answering(self, request):
        """
        Span-based question answering for a given question and context.
        We expect the input to use the (question, context) format for the text pairs.

        Args:
          request: the prediction request

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
                desired_spans = np.isin(starts_, undesired_tokens_.nonzero()) & np.isin(ends_,
                                                                                        undesired_tokens_.nonzero())
                starts_ = starts_[desired_spans]
                ends_ = ends_[desired_spans]
                scores_ = candidates[0, starts_, ends_]

                return starts_, ends_, scores_

        predictions, features = self._predict(request)
        # print(predictions, request)
        task_outputs = {"answers": []}
        for idx, (start, end, (_, context)) in enumerate(zip(predictions["start_logits"],
                                                             predictions["end_logits"], request)):
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

            # print(start, end)

            # Get score for 'no answer' then mask for decoding step (CLS token
            no_answer_score = (start[0] * end[0]).item()
            start[0] = end[0] = 0.0

            starts, ends, scores = decode(
                start, end, 1, 128, undesired_tokens
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
                              enc.word_to_chars(enc.token_to_word(e), sequence_index=1)[1]].lower(),
                }
                for s, e, score in zip(starts, ends, scores)]
            answers.append({"score": no_answer_score, "start": 0, "end": 0, "answer": ""})
            answers = sorted(answers, key=lambda x: x["score"], reverse=True)[:1]
            task_outputs["answers"].append(answers)
        return task_outputs


if __name__ == '__main__':
    base_model = "bert-base-uncased"
    adapter_model = "AdapterHub/bert-base-uncased-pf-squad_v2"
    model = BertAdapterModel.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    adapter_name = model.load_adapter(adapter_model, source="hf")
    model.active_adapters = adapter_name

    question, context = "Who patronized the monks in Italy?", \
                "At Saint Evroul, a tradition of singing had developed and the choir achieved fame in Normandy. Under the Norman abbot Robert de Grantmesnil, several monks of Saint-Evroul fled to southern Italy, where they were patronised by Robert Guiscard and established a Latin monastery at Sant'Eufemia. There they continued the tradition of singing."
    reduce = InputReduction(model,
                            tokenizer,
                            request={
                                "question": question,
                                "context": context,
                                "top_k": 5,
                                "mode": "max"
                            },
                            method="scaled_attn")
    resp = reduce.get_predictions()
    print(resp)

    # dataloader = PreprocessData("squad_v2", "squad_v2", save_data=False, save_path="../../")
    # outputs = list()
    # count = 0
    # for ex in tqdm(dataloader.processed_val_set()):
    #     # print(ex)
    #     result = dict()
    #     reduce = InputReduction(model,
    #                             tokenizer,
    #                             request={
    #                                 "question": ex["question"],
    #                                 "context": ex["context"],
    #                                 "top_k": 5,
    #                                 "mode": "max"
    #                             },
    #                             method="shap")
    #     resp = reduce.get_predictions()
    #     print(resp)
    #     count += 1
    #     if count == 2:
    #         break
