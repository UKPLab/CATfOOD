import string
import numpy as np
from abc import ABC
from typing import List, Tuple, Dict, Optional

import torch
from torch import backends
from torch.nn import Module, ModuleList
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


class BaseExplainer(ABC):
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
            self.tokenizer.sep_token_id
            if self.tokenizer.sep_token_id is not None
            else self.tokenizer.eos_token_id
        )
        self.cls_token_id = (
            self.tokenizer.cls_token_id
            if self.tokenizer.cls_token_id is not None
            else self.tokenizer.bos_token_id
        )

        self.model_prefix = model.base_model_prefix

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.word_embeddings = self.model.get_input_embeddings()
        self.position_embeddings = None
        self.token_type_embeddings = None

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

    def get_model_embeddings(self,
                             embedding_type: str = "word_embeddings"
                             ) -> Module or ModuleList:
        """
        Get the model embedding layer
        Args:
            embedding_type: can be one of word_embeddings,
                token_type_embeddings or position_embeddings
        """
        embeddings = Module or ModuleList
        model_prefix = self.model.base_model_prefix
        model_base = getattr(self.model, model_prefix)
        model_embeddings = getattr(model_base, "embeddings")
        if embedding_type == "word_embeddings":
            embeddings = model_embeddings.word_embeddings
        elif embedding_type == "token_type_embeddings":
            embeddings = model_embeddings.token_type_embeddings
        elif embedding_type == "position_embeddings":
            embeddings = model_embeddings.position_embeddings
        return embeddings

    def _register_hooks(self, embeddings_list: List, alpha: int):
        """
        Register the model embeddings during the forward pass
        Args:
            embeddings_list: list to store embeddings during forward pass
        """

        def forward_hook(module, inputs, output):
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).clone().detach())

        handles = []
        embedding_layer = self.get_model_embeddings()
        handles.append(embedding_layer.register_forward_hook(forward_hook))
        return handles

    def _register_embedding_gradient_hooks(self, embedding_grads: List):
        """
        Register the model gradients during the backward pass
        Args:
            embedding_grads: list to store the gradients
        """
        def hook_layers(module, grad_in, grad_out):
            grads = grad_out[0]
            embedding_grads.append(grads)

        hooks = []
        embedding_layer = self.get_model_embeddings()
        hooks.append(embedding_layer.register_full_backward_hook(hook_layers))
        return hooks

    def get_gradients(self, inputs, answer_start, answer_end):
        """
        Compute model gradients
        Args:
            inputs: list of question and context
            answer_start: answer span start
            answer_end: answer span end
        Return:
            dict of model gradients
        """
        embedding_gradients: List[torch.Tensor] = []
        # print(answer_start, answer_end)

        original_param_name_to_requires_grad_dict = {}
        for param_name, param in self.model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True

        hooks: List = self._register_embedding_gradient_hooks(embedding_gradients)
        with backends.cudnn.flags(enabled=False):
            encoded_inputs = self.encode(inputs, return_tensors="pt")
            encoded_inputs.to(self.device)
            outputs = self.model(
                **encoded_inputs,
                start_positions=answer_start.to(self.device),
                end_positions=answer_end.to(self.device)
            )
            loss = outputs.loss

            # Zero gradients.
            # NOTE: this is actually more efficient than calling `self._model.zero_grad()`
            # because it avoids a read op when the gradients are first updated below.
            for p in self.model.parameters():
                p.grad = None
            loss.backward()

        for hook in hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = "grad_input_" + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        # restore the original requires_grad values of the parameters
        for param_name, param in self.model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]
        # print(grad_dict)
        return grad_dict

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
        encoded_inputs = self.encode(inputs,
                                     add_special_tokens=True,
                                     return_tensors="pt")
        encoded_inputs.to(self.device)
        self.decoded_text = self.decode(
            encoded_inputs["input_ids"],
            skip_special_tokens=False
        )
        self.words_mapping = encoded_inputs.word_ids()

        all_predictions = list()
        self.model.to(self.device)
        predictions = self.model(
            **encoded_inputs,
            **model_kwargs
        )
        # print(predictions)
        all_predictions.append(predictions)
        keys = all_predictions[0].keys()
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

        return predictions, encoded_inputs

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
            desired_spans = np.isin(starts_, undesired_tokens_.nonzero()) \
                            & np.isin(ends_, undesired_tokens_.nonzero())
            starts_ = starts_[desired_spans]
            ends_ = ends_[desired_spans]
            scores_ = candidates[0, starts_, ends_]

            return starts_, ends_, scores_

        predictions, features = self._predict(request)

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
                              enc.word_to_chars(enc.token_to_word(e), sequence_index=1)[1]],
                }
                for s, e, score in zip(starts, ends, scores)]
            answers.append({"score": no_answer_score, "start": 0, "end": 0, "answer": ""})
            answers = sorted(answers, key=lambda x: x["score"], reverse=True)[:1]
            task_outputs["answers"].append(answers)
        return task_outputs

    def handle_edge_cases(self, tokens):
        context_start = tokens.index(self.tokenizer.eos_token)
        word_map = []
        # print(context_start)
        # force_break = False
        for i, t in enumerate(tokens):
            # special token
            if t in self.tokenizer.all_special_tokens:
                word_map.append(None)
                continue
            if i < context_start and (t == "'s" or t == "'t" or t == "th"):
                word_map.append(word_map[i - 1])
            elif i < context_start:
                if len(word_map) == 1:
                    word_map.append(0)
                else:
                    word_map.append(word_map[i - 1] + 1)
            elif i > context_start and (t == "'s" or t == "'t" or t == "th"):
                word_map.append(word_map[i - 1])
            elif i > context_start:
                if word_map[i - 1] is None:
                    word_map.append(0)
                else:
                    word_map.append(word_map[i - 1] + 1)
        return word_map

    def _bpe_decode(
            self,
            tokens: List[str],
            # attributions: List
    ) -> List:

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

        # filtered_tokens, scores = [], []
        # for s0, s1 in segments:
        #     filtered_tokens.append(''.join(decoded_each_tok[s0:s1]))
        #     # scores.append(np.sum(attributions[s0:s1], axis=0))
        # filtered_tokens = [token.lstrip() for token in filtered_tokens]
        # attribution_score = np.stack(scores, axis=0)

        return segments  # attribution_score

    def _wordpiece_decode(
            self,
            tokens: List[str],
            attributions: List
    ) -> Tuple[List[str], np.array]:

        decoded_each_tok = tokens
        word_map = self.words_mapping
        chars_to_handle = ["s", "t", "ve", "re", "m", "n't"]

        context_start = tokens.index(self.tokenizer.sep_token)
        for idx, token in enumerate(decoded_each_tok[:-1]):
            if token not in self.tokenizer.all_special_tokens \
                    and token == "'" \
                    and decoded_each_tok[idx+1] in chars_to_handle \
                    and idx < context_start:
                word_map[idx] = word_map[idx-1]
                word_map[idx+1] = word_map[idx-1]
                word_map[idx+2:context_start] = [w-2 for w in word_map[idx+2:context_start] if w]
                continue
            if token not in self.tokenizer.all_special_tokens \
                    and token == "'" \
                    and decoded_each_tok[idx+1] in chars_to_handle \
                    and idx > context_start:
                word_map[idx] = word_map[idx-1]
                word_map[idx+1] = word_map[idx-1]
                word_map[idx+2:-1] = [w-2 for w in word_map[idx+2:-1] if w]
                continue

        filtered_tokens = [decoded_each_tok[0]]
        for idx, (word_idx, word) in enumerate(zip(word_map, decoded_each_tok[1:])):
            if word_idx == word_map[idx + 1] and not word == self.tokenizer.sep_token:
                filtered_tokens[-1] = f'{filtered_tokens[-1]}{word.replace("##", "")}'
            else:
                filtered_tokens.append(word)

        attribution_score = [attributions[0]]
        for idx, (word_idx, score) in enumerate(zip(word_map, attributions[1:])):
            if word_idx == word_map[idx + 1] and word_idx is not None:
                attribution_score[-1] = attribution_score[-1] + score
            else:
                attribution_score.append(score)

        return filtered_tokens, np.array(attribution_score)

    def process_outputs(self, attributions: List, top_k: int, mode: str) -> List[Dict]:
        """
        post-process the word attributions to merge the sub-words tokens
        to words
        Args:
            attributions: word importance scores
            top_k: number of top word attributions
            mode: whether to show attribution in question, context or both
        Returns:
            dict of processed words along with their scores
        """

        filtered_tokens: list = []
        importance: np.array = np.array([])
        sep_tokens: int = 0
        dec_text = self.decoded_text
        if self.model.config.model_type in ["roberta", "bart"]:
            filtered_tokens, importance = self._bpe_decode(dec_text, attributions)
            sep_tokens = 2
        elif self.model.config.model_type == "bert":
            filtered_tokens, importance = self._wordpiece_decode(dec_text, attributions)
            sep_tokens = 1

        normed_imp = [np.round(float(i) / sum(importance), 3)
                      for i in importance]
        result = [(w, a) for w, a in zip(filtered_tokens, normed_imp)
                  if w != '']
        assert len(filtered_tokens) == len(normed_imp)
        # outputs = {"attributions": result}
        context_start = filtered_tokens.index(self.tokenizer.sep_token)
        # account for cls token in result
        question = [(idx, v[0], v[1])
                    for idx, v in enumerate(result[1:])
                    if idx < context_start - 1]

        context = [(idx - len(question) - sep_tokens, v[0], v[1])
                   for idx, v in enumerate(result[1:])
                   if idx > context_start - 1
                   and v[0] != self.tokenizer.sep_token]

        outputs, outputs_question, outputs_context = [], [], []
        if mode == "question" or mode == "all":
            outputs_question = [(i, k.lower(), v) for i, k, v in sorted(
                question,
                key=lambda item: item[2],
                reverse=True)[:top_k]]
        if mode == "context" or mode == "all":
            outputs_context = [(i, k.lower(), v) for i, k, v in sorted(
                context,
                key=lambda item: item[2],
                reverse=True)[:top_k]]

        outputs = [{"question": outputs_question, "context": outputs_context}]
        return outputs
