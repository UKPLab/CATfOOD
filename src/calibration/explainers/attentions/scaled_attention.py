import math
import numpy as np
from tqdm import tqdm
import traceback
from collections import OrderedDict
from transformers import (
    AutoTokenizer,
    # BertAdapterModel,
    RobertaForQuestionAnswering,
    # RobertaAdapterModel,
    PreTrainedModel,
    PreTrainedTokenizer,
    logging
)

import torch
from torch import backends
from torch.nn import Module, ModuleList
from typing import List

from src.calibration.explainers.base_explainer import BaseExplainer
from src.calibration.baseline import dataloader, utils

logging.set_verbosity_error()

torch.manual_seed(4)
torch.cuda.manual_seed(4)
np.random.seed(4)

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"

class ScaledAttention(BaseExplainer):
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer
                 ):
        super().__init__(model=model, tokenizer=tokenizer)

    def get_model_attentions(self) -> Module or ModuleList:
        """
        Get the model attention layer
        :return:
        """
        model_prefix = self.model.base_model_prefix
        model_base = getattr(self.model, model_prefix)
        model_enc = getattr(model_base, "encoder")
        # get attn weights from last layer
        attentions = model_enc.layer[-1].attention
        return attentions

    def _register_forward_hooks(self, attentions_list: List):
        """
        Register the model attentions during the forward pass
        :param attentions_list:
        :return:
        """

        def forward_hook(module, inputs, output):
            attentions_list.append(output[1][:, :, 0, :].mean(1).squeeze(0).clone().detach())

        handles = []
        attn_layer = self.get_model_attentions()
        handles.append(attn_layer.register_forward_hook(forward_hook))
        return handles

    def _register_attention_gradient_hooks(self, attn_grads: List):
        """
        Register the model gradients during the backward pass
        :param embedding_grads:
        :return:
        """
        def hook_layers(module, grad_in, grad_out):
            grads = grad_out[0]
            attn_grads.append(grads)

        hooks = []
        attentions = self.get_model_attentions()
        hooks.append(attentions.register_full_backward_hook(hook_layers))
        return hooks

    def get_gradients(self, inputs, answer_start, answer_end):
        """
        Compute model gradients
        :param inputs: list of question and context
        :param answer_start: answer span start
        :param answer_end: answer span end
        :return: dict of model gradients
        """
        attn_gradients: List[torch.Tensor] = []
        # print(answer_start, answer_end)

        original_param_name_to_requires_grad_dict = {}
        for param_name, param in self.model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True

        hooks: List = self._register_attention_gradient_hooks(attn_gradients)
        with backends.cudnn.flags(enabled=False):
            encoded_inputs = self.encode(inputs, return_tensors="pt")
            encoded_inputs.to(self.device)
            outputs = self.model(
                **encoded_inputs,
                start_positions=answer_start,
                end_positions=answer_end,
                output_attentions=True
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
        for idx, grad in enumerate(attn_gradients):
            key = "grad_input_" + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        # restore the original requires_grad values of the parameters
        for param_name, param in self.model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]
        # print(grad_dict)
        return grad_dict

    def interpret(self, inputs: List[List]):
        # get predicted answer
        outputs, _ = self._predict(inputs)
        start_idx = torch.argmax(outputs[0])
        end_idx = torch.argmax(outputs[1])

        answer_start = torch.tensor([start_idx], device=self.device)
        answer_end = torch.tensor([end_idx], device=self.device)

        attentions_list: List[torch.Tensor] = []
        # Hook used for saving embeddings
        handles: List = self._register_forward_hooks(attentions_list)
        try:
            grads = self.get_gradients(inputs, answer_start, answer_end)
        finally:
            for handle in handles:
                handle.remove()

        # Gradients come back in the reverse order that they were sent into the network
        attentions_list.reverse()
        attentions_list = [attn.cpu().detach().numpy() for attn in attentions_list]

        instances_with_grads = dict()
        for key, grad in grads.items():
            input_idx = int(key[-1]) - 1
            attn_grad = np.sum(grad[0] * attentions_list[input_idx][0], axis=1)
            norm = np.linalg.norm(attn_grad, ord=1)
            normalized_grad = [math.fabs(e) / norm for e in attn_grad]
            grads[key] = normalized_grad

        instances_with_grads["instance_" + str(1)] = grads

        outputs = {"attributions": instances_with_grads["instance_1"]["grad_input_1"]}
        return outputs


if __name__ == '__main__':
    # load model
    import argparse

    parser = argparse.ArgumentParser(description="Passing arguments for model, tokenizer, and dataset.")

    parser.add_argument(
        "--model_name",
        default="",
        type=str, required=False, help="Specify the model to use.")
    parser.add_argument("--tokenizer", default="roberta-base", type=str, required=False,
                        help="Specify the tokenizer to use.")
    parser.add_argument("--dataset", type=str, required=True, help="Specify the dataset to use.")

    args = parser.parse_args()

    model = RobertaForQuestionAnswering.from_pretrained(BASE_PATH + args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.dataset == "squad":
        loader = dataloader.PreprocessData("squad", "plain_text", save_data=False, save_path="../../../../../")
        data = loader.processed_val_set()
    elif args.dataset == "squad_adversarial":
        loader = dataloader.PreprocessData("squad_adversarial", "AddSent", save_data=False, save_path="../../../../../")
        data = loader.processed_val_set()
    elif args.dataset == "trivia_qa":
        data = dataloader.get_dev_examples("./src/data", "dev_trivia.json")
    elif args.dataset == "hotpot_qa":
        data = dataloader.get_dev_examples("./src/data", "dev_hotpot.json")
    elif args.dataset == "news_qa":
        data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/NewsQA.jsonl")
    elif args.dataset == "bioasq":
        data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/BioASQ-dev.jsonl")
    elif args.dataset == "natural_questions":
        data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/NaturalQuestionsShort.jsonl")
    else:
        raise ValueError("Dataset not supported.")

    outputs = list()
    grads = ScaledAttention(model=model, tokenizer=tokenizer)
    c = 0
    processed_instances = OrderedDict()
    if args.dataset == "squad_adversarial":
        for ex in tqdm(data):
            try:
                # if ex["id"] == "56f879bdaef23719006260e2":
                scores = grads.interpret([[ex["question"], ex["context"]]])
                processed_instances[ex["id"]] = scores
                c += 1
            except Exception:
                print(ex)
                print(f"Unable to get attributions: {traceback.format_exc()}")
    elif args.dataset in ["trivia_qa", "hotpot_qa", "news_qa", "natural_questions", "bioasq"]:
        def remove_white_space(example):
            example["question_text"] = ' '.join(example["question_text"].split())
            example["context_text"] = ' '.join(example["context_text"].split())
            return example

        # data = dataloader.get_dev_examples(BASE_PATH+"src/data", "dev_hotpot.json")
        # data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/NewsQA.jsonl")
        # data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/BioASQ-dev.jsonl")
        # data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/NaturalQuestionsShort.jsonl")

        for ex in tqdm(data):
            ex = remove_white_space(ex)
            try:
                # if ex["id"] == "56f879bdaef23719006260e2":
                scores = grads.interpret([[ex["question_text"], ex["context_text"]]])
                processed_instances[ex["qas_id"]] = scores
                c += 1
            except Exception:
                print(ex)
                print(f"Unable to get attributions: {traceback.format_exc()}")

    print(f"Processed {c} instances of original data")
    utils.dump_to_bin(processed_instances,
                      BASE_PATH + f"src/data/{args.dataset}/sc_attn_info_{args.model_name}.bin")
    print(f"Saved instances: {c}")