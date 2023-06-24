import math
from tqdm import tqdm
from collections import OrderedDict
import traceback
import numpy as np
from typing import List
import torch
from transformers import (
    AutoTokenizer,
    # BertAdapterModel,
    # RobertaAdapterModel,
    # AutoModelWithHeads,
    RobertaForQuestionAnswering,
    PreTrainedModel,
    PreTrainedTokenizer,
    logging
)
from src.calibration.explainers.base_explainer import BaseExplainer
from src.calibration.baseline import dataloader, utils

logging.set_verbosity_error()

torch.manual_seed(4)
torch.cuda.manual_seed(4)
np.random.seed(4)

BASE_PATH = "/storage/xyz/work/anon/research_projects/exp_calibration/"


class SimpleGradients(BaseExplainer):
    """
    class for the implementation of simple gradients' explanation method
    """
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer
                 ):
        super().__init__(model=model, tokenizer=tokenizer)

    def interpret(self,
                  inputs: List[List],
                  ):
        """
        gets the word attributions
        """
        # get predicted answer
        outputs, _ = self._predict(inputs)
        start_idx = torch.argmax(outputs[0])
        end_idx = torch.argmax(outputs[1])

        answer_start = torch.tensor([start_idx])
        answer_end = torch.tensor([end_idx])

        embeddings_list: List[torch.Tensor] = []
        # Hook used for saving embeddings
        handles: List = self._register_hooks(embeddings_list, alpha=0)
        try:
            grads = self.get_gradients(inputs, answer_start, answer_end)
        finally:
            for handle in handles:
                handle.remove()

        # Gradients come back in the reverse order that they were sent into the network
        embeddings_list.reverse()
        embeddings_list = [embedding.cpu().detach().numpy() for embedding in embeddings_list]
        # token_offsets.reverse()
        # embeddings_list = self._aggregate_token_embeddings(embeddings_list, token_offsets)
        instances_with_grads = dict()
        for key, grad in grads.items():
            # Get number at the end of every gradient key (they look like grad_input_[int],
            # we're getting this [int] part and subtracting 1 for zero-based indexing).
            # This is then used as an index into the reversed input array to match up the
            # gradient and its respective embedding.
            input_idx = int(key[-1]) - 1
            # The [0] here is undo-ing the batching that happens in get_gradients.
            emb_grad = np.sum(grad[0] * embeddings_list[input_idx][0], axis=1)
            norm = np.linalg.norm(emb_grad, ord=1)
            normalized_grad = [math.fabs(e) / norm for e in emb_grad]
            grads[key] = normalized_grad

        instances_with_grads["instance_" + str(1)] = grads

        outputs = {"attributions": instances_with_grads["instance_1"]["grad_input_1"]}

        return outputs


if __name__ == '__main__':

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

    grads = SimpleGradients(model=model, tokenizer=tokenizer)
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
                      BASE_PATH + f"src/data/{args.dataset}/simple_grads_base.bin")
    print(f"Saved instances: {c}")
