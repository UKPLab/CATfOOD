from typing import List
from tqdm import tqdm
from collections import OrderedDict
import traceback

from transformers import (
    AutoTokenizer,
    # BertAdapterModel,
    # RobertaAdapterModel,
    RobertaForQuestionAnswering,
    PreTrainedModel,
    PreTrainedTokenizer,
    logging
)

from src.calibration.explainers.base_explainer import BaseExplainer
from src.calibration.baseline import dataloader, utils

logging.set_verbosity_error()

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"

class AttnAttribution(BaseExplainer):
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer
                 ):
        super().__init__(model=model, tokenizer=tokenizer)

    def interpret(self, inputs: List[List]):
        # get predicted answer
        model_kwargs = {"output_attentions": True}
        outputs, _ = self._predict(inputs, **model_kwargs)
        attn = outputs["attentions"][-1]
        weights = attn[:, :, 0, :].mean(1)
        attributions = weights.cpu().detach().numpy()[0]
        outputs = {"attributions": attributions}

        return outputs


if __name__ == '__main__':
    # model_path = BASE_PATH+"roberta-squad-flan-ul2-v1-temp-0.7"
    # model = RobertaForQuestionAnswering.from_pretrained(model_path)
    # tokenizer = AutoTokenizer.from_pretrained("roberta-base")
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
    grads = AttnAttribution(model=model, tokenizer=tokenizer)
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
                      BASE_PATH + f"src/data/{args.dataset}/attn_info_base.bin")
    print(f"Saved instances: {c}")

    # # process counterfactuals
    # cf_path = BASE_PATH + "src/data/squad_adversarial/rag_counterfactuals_turk0_last.jsonl"
    # c = 0
    # with jsonlines.open(cf_path) as reader:
    #     for ex in tqdm(reader):
    #         try:
    #             scores = grads.interpret([[ex["question"], ex["context"]]])
    #             processed_instances[ex["id"].replace("_", "-cf-")] = scores
    #             c += 1
    #         except Exception as e:
    #             print(f"Unable to get cf attributions: {e}")
    #             print(ex)
    #
    #     print(f"Processed {c} instances of counterfactual data")

    # print(processed_instances)
    # print(ast.literal_eval(processed_instances["56be4db0acb8001400a502ee"]))
    # print(len(ast.literal_eval(processed_instances["56be4db0acb8001400a502ee"])["attributions"][0]))



    ### Trivia QA

    # print(f"Processed {c} instances of original data")

    # # process counterfactuals
    # cf_path = BASE_PATH + "src/data/natural_questions/rag_counterfactuals_turk0.jsonl"
    # c = 0
    # with jsonlines.open(cf_path) as reader:
    #     for ex in tqdm(reader):
    #         try:
    #             scores = grads.interpret([[ex["question"], ex["context"]]])
    #             processed_instances[ex["id"].replace("_", "-cf-")] = scores
    #             c += 1
    #         except Exception as e:
    #             print(f"Unable to get cf attributions: {e}")
    #             print(ex)
    #
    #     print(f"Processed {c} instances of counterfactual data")
    #
    # utils.dump_to_bin(processed_instances,
    #                   BASE_PATH + "src/data/trivia_qa/attn_info_flan_ul2_cleaned.bin")
    #
    # print(f"Saved instances: {c}")
