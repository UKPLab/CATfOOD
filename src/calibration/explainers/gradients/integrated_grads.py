import math
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Any
from collections import OrderedDict

from transformers import (
    AutoTokenizer,
    # BertAdapterModel,
    # RobertaAdapterModel,
    RobertaForQuestionAnswering,
    PreTrainedModel,
    PreTrainedTokenizer,
    logging,
)

from src.calibration.explainers.base_explainer import BaseExplainer
from src.calibration.baseline import dataloader, utils

import traceback

import torch

logging.set_verbosity_error()

torch.manual_seed(4)
torch.cuda.manual_seed(4)
np.random.seed(4)

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"


class IntegratedGradients(BaseExplainer):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        super().__init__(model=model, tokenizer=tokenizer)

    def _register_hooks(self, embeddings_list: List, alpha: int):
        def forward_hook(module, inputs, output):
            # Save the input for later use. Only do so on first call.
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).clone().detach())
            # Scale the embedding by alpha
            output.mul_(alpha)

        handles = []
        embedding_layer = self.get_model_embeddings()
        handles.append(embedding_layer.register_forward_hook(forward_hook))
        return handles

    def _integrate_gradients(
        self, inputs: List[List]
    ) -> Tuple[Dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
        """
        Returns:
             integrated gradients for the given [`Instance`]
        """
        ig_grads: Dict[str, Any] = {}

        # List of Embedding inputs
        embeddings_list: List[torch.Tensor] = []

        # answer prediction
        outputs, _ = self._predict(inputs)
        start_idx = torch.argmax(outputs[0])
        end_idx = torch.argmax(outputs[1])
        answer_start = torch.tensor([start_idx])
        answer_end = torch.tensor([end_idx])

        # Use 10 terms in the summation approximation of the integral in integrated grad
        steps = 10
        # Exclude the endpoint because we do a left point integral approximation
        for alpha in np.linspace(0, 1.0, num=steps, endpoint=False):
            handles = []
            # Hook for modifying embedding value
            handles = self._register_hooks(embeddings_list, alpha)

            try:
                grads = self.get_gradients(inputs, answer_start, answer_end)
            finally:
                for handle in handles:
                    handle.remove()

            # Running sum of gradients
            if ig_grads == {}:
                ig_grads = grads
            else:
                for key in grads.keys():
                    ig_grads[key] += grads[key]

        # Average of each gradient term
        for key in ig_grads.keys():
            ig_grads[key] /= steps

        # Gradients come back in the reverse order that they were sent into the network
        embeddings_list.reverse()
        embeddings_list = [
            embedding.cpu().detach().numpy() for embedding in embeddings_list
        ]
        # Element-wise multiply average gradient by the input
        for idx, input_embedding in enumerate(embeddings_list):
            # print(idx, input_embedding)
            key = "grad_input_" + str(idx + 1)
            ig_grads[key] *= input_embedding

        return ig_grads, answer_start, answer_end

    def interpret(self, inputs: List[List]):
        # run integrated grad
        grads, answer_start, answer_end = self._integrate_gradients(inputs)
        # normalize results
        instances_with_grads = dict()
        for key, grad in grads.items():
            # The [0] here is undo-ing the batching that happens in get_gradients.
            embedding_grad = np.sum(grad[0], axis=1)
            norm = np.linalg.norm(embedding_grad, ord=1)
            normalized_grad = np.array([math.fabs(e) / norm for e in embedding_grad])
            grads[key] = normalized_grad

        instances_with_grads["instance_" + str(1)] = grads
        outputs = {"attributions": instances_with_grads["instance_1"]["grad_input_1"]}
        # print(len(instances_with_grads["instance_1"]["grad_input_1"]))

        return outputs


if __name__ == "__main__":
    # model_path = BASE_PATH + "roberta-squad-flan-ul2-v1-temp-0.7"
    # model = RobertaForQuestionAnswering.from_pretrained(model_path)
    # tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    import argparse

    parser = argparse.ArgumentParser(
        description="Passing arguments for model, tokenizer, and dataset."
    )

    parser.add_argument(
        "--model_name",
        default="",
        type=str,
        required=False,
        help="Specify the model to use.",
    )
    parser.add_argument(
        "--tokenizer",
        default="roberta-base",
        type=str,
        required=False,
        help="Specify the tokenizer to use.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Specify the dataset to use."
    )

    args = parser.parse_args()

    model = RobertaForQuestionAnswering.from_pretrained(BASE_PATH + args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.dataset == "squad":
        loader = dataloader.PreprocessData(
            "squad", "plain_text", save_data=False, save_path="../../../../../"
        )
        data = loader.processed_val_set()
    elif args.dataset == "squad_adversarial":
        loader = dataloader.PreprocessData(
            "squad_adversarial", "AddSent", save_data=False, save_path="../../../../../"
        )
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
        data = dataloader.get_dev_samples_mrqa(
            BASE_PATH + "src/data/NaturalQuestionsShort.jsonl"
        )
    else:
        raise ValueError("Dataset not supported.")

    outputs = list()

    grads = IntegratedGradients(model=model, tokenizer=tokenizer)
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
    elif args.dataset in [
        "trivia_qa",
        "hotpot_qa",
        "news_qa",
        "natural_questions",
        "bioasq",
    ]:

        def remove_white_space(example):
            example["question_text"] = " ".join(example["question_text"].split())
            example["context_text"] = " ".join(example["context_text"].split())
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
    utils.dump_to_bin(
        processed_instances,
        BASE_PATH + f"src/data/{args.dataset}/ig_info_{args.model_name}.bin",
    )
    print(f"Saved instances: {c}")
