"""
File to run SHAP interpretation
"""

import logging
import os
import shutil
import random
import timeit
import collections
import copy

import numpy as np
import torch
from typing import List, Tuple
from tqdm import tqdm
import pandas as pd

from datasets import Dataset
from common.config import InterpConfig, load_config_and_tokenizer

import dataloader

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
)

from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids
from transformers import PreTrainedTokenizer, PreTrainedModel
from data.qa_metrics import squad_evaluate

from probe.probe_models import ProbeRobertaForQuestionAnswering
from vis_tools.vis_utils import visualize_token_attributions

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"
# BASE_PATH = "/home/sachdeva/projects/ukp/exp_calibration/"

import itertools
from functools import partial
from utils import run_shap_attribution


def _mkdir_f(prefix):
    if os.path.exists(prefix):
        shutil.rmtree(prefix)
    os.makedirs(prefix)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def ig_analyze(args, tokenizer):
    filenames = os.listdir(args.interp_dir)
    # print(filenames)
    filenames = filenames[:2]
    _mkdir_f(args.visual_dir)
    for fname in tqdm(filenames, desc='Visualizing'):
        interp_info = torch.load(os.path.join(args.interp_dir, fname))
        # datset_stats.append(stats_of_ig_interpretation(tokenizer, interp_info))
        visualize_token_attributions(args, tokenizer, interp_info, fname)


class ShapLM:
    def __init__(
        self,
        args,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        self.args =args
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

        self.model_prefix = self.model.base_model_prefix
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

        features = self.encode(
            request,
            add_special_tokens=True,
            return_tensors="pt",
        )

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

        task_outputs = {"answers": [], "answer_start": [], "answer_end": []}
        for idx, (start, end, (_, context)) in enumerate(zip(predictions["start_logits"],
                                                             predictions["end_logits"],
                                                             request)):

            start_idx = torch.argmax(start)
            end_idx = torch.argmax(end)
            answer_start = torch.tensor([start_idx])
            answer_end = torch.tensor([end_idx])

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
            task_outputs["answer_start"].extend(answer_start.cpu().detach().numpy())
            task_outputs["answer_end"].extend(answer_end.cpu().detach().numpy())
        return predictions, features, task_outputs

    def predict_with_mask(self, active_mask, tokenizer, model, base_inputs, answers, full_input_ids):
        input_ids = tokenizer.mask_token_id * torch.ones_like(full_input_ids)
        input_ids[0, active_mask == 1] = full_input_ids[0, active_mask == 1]
        prob = model.probe_forward(**base_inputs, input_ids=input_ids, answers=answers)
        return prob

    def run_shap(self, features, answers):

        # make a copy of the input features
        input_features = copy.deepcopy(features)
        input_features = input_features.to(self.device)
        tokens = input_features.tokens()
        input_features['return_kl'] = False

        full_input_ids = input_features.pop('input_ids')
        full_position_ids = create_position_ids_from_input_ids(full_input_ids, self.tokenizer.pad_token_id).to(
            full_input_ids.device)

        # fix position id
        input_features['position_ids'] = full_position_ids
        # fix cls ? maybe
        score_fn = partial(self.predict_with_mask, tokenizer=self.tokenizer, model=self.model,
                           base_inputs=input_features,
                           answers=answers,
                           full_input_ids=full_input_ids)

        np_attribution = run_shap_attribution(self.args, len(tokens), score_fn).reshape((1, -1))
        return torch.from_numpy(np_attribution)

    def shap_interp(self, prefix=""):
        args = self.args
        if not os.path.exists(args.interp_dir):
            os.makedirs(args.interp_dir)

        # fix the model
        self.model.requires_grad_(False)

        def group_batch(batch):
            return {k: [v] for k, v in batch.items()}

        # data = dataloader.PreprocessData(
        #     args.dataset,
        #     args.dataset_config,
        #     save_data=False,
        #     save_path="../../"
        # )
        # dataset = data.processed_val_set()
        # dataset = dataset.select(range(1445, 3560))

        # trivia_qa
        # dataset = Dataset.from_generator(dataloader.get_dev_examples_hf)
        # dataset = dataset.select(range(0, 3000))
        # dataset = dataset.filter(lambda example: example['id'] == "")

        # for shortcuts
        # dataloader = token_in_context.Shortcut(
        #     args.dataset,
        #     args.dataset_config,
        #     args.percent,
        #     args.percent_augment
        # )
        # _, dataset = dataloader.create_synthetic_set()

        # dataset = Dataset.from_generator(dataloader.get_dev_examples_hf)
        # dataset = dataset.select(range(0, 618))

        # MRQA datasets
        ##############################################################################################
        data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/BioASQ-dev.jsonl")
        data = data[1298:]

        # Define a dictionary to map old keys to new keys
        key_mapping = {
            "qas_id": "id",
            "question_text": "question",
            "context_text": "context",
            "answer_text": "answers",
            # Add more key mappings...
        }

        # Rename specified keys in each sample
        # Rename specified keys and wrap values in lists in each sample
        renamed_samples = []
        for sample in data:
            renamed_sample = {
                new_key: [sample[old_key]] if old_key == "answer_text" else sample[old_key]
                for old_key, new_key in key_mapping.items()
            }
            renamed_samples.append(renamed_sample)

        df = pd.DataFrame(renamed_samples)
        # Create a pseudo-dataset with the list
        dataset = Dataset.from_pandas(df)

        ################################################################################################

        eval_dataloader = dataset.map(
            group_batch, batched=True, batch_size=args.eval_batch_size
        )

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_predictions = []
        start_time = timeit.default_timer()

        for idx, batch in tqdm(enumerate(eval_dataloader), desc="Interpreting",
                               total=min(len(dataset), args.first_n_samples)):
            if idx == args.first_n_samples:
                break
            # changes here for dataset
            predictions, features, answers = self.question_answering(
                request=[[ques, cxt] for ques, cxt in zip(batch["question"], batch["context"])],
            )

            # input_features = features
            with torch.no_grad():
                importances = self.run_shap(features, answers)

            answers["id"] = batch["id"]
            batch_predictions = collections.OrderedDict(answers)
            # print("batch_predictions", batch_predictions)

            self.dump_shap_info(args, batch, features, answers, importances)
            # lots of info, dump to files immediately
            all_predictions.append(batch_predictions)
            # break

        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

        # Compute the F1 and exact scores.
        def merge_predictions(dicts):
            return dict(itertools.chain(*[list(x.items()) for x in dicts]))
        all_predictions = merge_predictions(all_predictions)
        results = squad_evaluate(eval_dataloader[:len(all_predictions["id"])], all_predictions)
        return results

    def dump_shap_info(self, args, examples, features, predictions, attributions):
        """
        Save the SHAP attributions to a file.
        """
        for idx in range(len(examples["id"])):

            actual_len = len(features.tokens(idx))
            attribution = attributions[idx][:actual_len].clone().detach()
            example = [examples["question"][idx], examples["context"][idx], examples["answers"][idx]]

            filename = os.path.join(args.interp_dir, f'{examples["id"][idx]}.bin')
            prelim_result = [predictions["answer_start"][idx], predictions["answer_end"][idx]]
            prediction = predictions["answers"][idx]
            feature = [features.tokens(idx), features.input_ids[idx].numpy(), features.attention_mask[idx].numpy()]
            torch.save(
                {
                    'example': example,
                    'feature': feature,
                    'prediction': prediction,
                    'prelim_result': prelim_result,
                    'attribution': attribution
                },
                filename
            )


def main():
    args = InterpConfig()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config, tokenizer = load_config_and_tokenizer(args)
    # Set seed
    set_seed(args)

    if args.do_vis:
        ig_analyze(args, tokenizer)
    else:
        # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
        logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
        checkpoint = args.model_name_or_path
        logger.info("Evaluate the following checkpoints: %s", checkpoint)

        # Reload the model
        model = ProbeRobertaForQuestionAnswering.from_pretrained(checkpoint)
        model.to(args.device)

        interpret = ShapLM(args, model, tokenizer)
        # Evaluate
        result = interpret.shap_interp()
        logger.info("Results: {}".format(result))

        return result


if __name__ == "__main__":
    main()
