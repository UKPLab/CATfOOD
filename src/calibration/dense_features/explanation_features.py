from transformers import (
    AutoTokenizer,
    # BertAdapterModel,
    # RobertaAdapterModel,
    RobertaForQuestionAnswering
)
import torch
from tqdm import tqdm
from collections import OrderedDict
import traceback
from src.calibration.baseline import dataloader, utils

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"


class DenseRepresentations:
    def __init__(self, model, tokenizer):
        self.model= model
        self.tokenizer = tokenizer

    def extract_representations(self, inputs):

        input_ids = self.tokenizer.encode(
            inputs,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            )
        # Get the start and end logits
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[0]

        return hidden_states


if __name__ == '__main__':
    # extract_representations()

    import argparse

    parser = argparse.ArgumentParser(description="Passing arguments for model, tokenizer, and dataset.")

    parser.add_argument(
        "--model_name",
        default="roberta-squad-flan-ul2-context-rel-noise-seed-42",
        type=str, required=False, help="Specify the model to use.")
    parser.add_argument("--tokenizer", default="roberta-base", type=str, required=False,
                        help="Specify the tokenizer to use.")
    parser.add_argument("--dataset", type=str, required=True, help="Specify the dataset to use.")

    args = parser.parse_args()

    model = RobertaForQuestionAnswering.from_pretrained(BASE_PATH + args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.dataset == "squad":
        loader = dataloader.PreprocessData("squad", "plain_text", save_data=False, save_path="../../../../")
        data = loader.processed_val_set()
    elif args.dataset == "squad_adversarial":
        loader = dataloader.PreprocessData("squad_adversarial", "AddSent", save_data=False, save_path="../../../../")
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
    c = 0
    representation = DenseRepresentations(model=model, tokenizer=tokenizer)
    processed_instances = OrderedDict()

    if args.dataset == "squad_adversarial":
        for ex in tqdm(data):
            try:
                # if ex["id"] == "56f879bdaef23719006260e2":
                states = representation.extract_representations([ex["question"], ex["context"]])
                processed_instances[ex["id"]] = states
                c += 1
            except Exception:
                print(ex)
                print(f"Unable to get representations: {traceback.format_exc()}")
    elif args.dataset in ["trivia_qa", "hotpot_qa", "news_qa", "natural_questions", "bioasq"]:
        def remove_white_space(example):
            example["question_text"] = ' '.join(example["question_text"].split())
            example["context_text"] = ' '.join(example["context_text"].split())
            return example


        for ex in tqdm(data):
            ex = remove_white_space(ex)
            try:
                states = representation.extract_representations([ex["question_text"], ex["context_text"]])
                processed_instances[ex["qas_id"]] = states
                c += 1
            except Exception:
                print(ex)
                print(f"Unable to get attributions: {traceback.format_exc()}")

    print(f"Processed {c} instances of original data")
    utils.dump_to_bin(processed_instances,
                      BASE_PATH + f"src/data/{args.dataset}/dense_repr_info_rag.bin")
    print(f"Saved instances: {c}")
