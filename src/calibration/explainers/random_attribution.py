import json
import numpy as np
from typing import List, Dict
import random
from tqdm import tqdm
from collections import OrderedDict
import traceback

import torch
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

logging.set_verbosity_error()


class RandomAttributions(BaseExplainer):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        super().__init__(model=model, tokenizer=tokenizer)

    def interpret(self, inputs: List[List], output: str = "processed"):
        # get predicted answer
        outputs, _ = self._predict(inputs)
        start_idx = torch.argmax(outputs[0])
        end_idx = torch.argmax(outputs[1])

        answer_start = torch.tensor([start_idx])
        answer_end = torch.tensor([end_idx])

        attributions = [random.uniform(0, 1) for i in range(outputs[0].size()[1])]

        if output == "raw":
            return (
                self.encode(inputs, add_special_tokens=True, return_tensors="pt"),
                attributions,
                answer_start,
                answer_end,
            )

        outputs = self.process_outputs(attributions=attributions)

        return json.dumps(outputs, indent=4)

    def process_outputs(self, attributions: List, top_k=10, mode="all") -> Dict:
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

        segments: list = []
        importance: np.array = np.array([])
        dec_text = self.decoded_text
        if self.model.config.model_type in ["roberta", "bart"]:
            segments = self._bpe_decode(dec_text)  # , attributions)
        elif self.model.config.model_type == "bert":
            filtered_tokens, importance = self._wordpiece_decode(dec_text, attributions)

        # normed_imp = [np.round(float(i) / sum(attributions), 3)
        #               for i in attributions]
        # result = [(w, a) for w, a in zip(filtered_tokens, normed_imp)]
        # if w != '']
        # assert len(segments) == len(attributions)
        # print(len(result))
        outputs = {"attributions": attributions, "segments": segments}
        return outputs


if __name__ == "__main__":
    # base_model = "bert-base-uncased"
    # adapter_model = "AdapterHub/bert-base-uncased-pf-squad_v2"
    # model = BertAdapterModel.from_pretrained(base_model)
    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    # adapter_name = model.load_adapter(adapter_model, source="hf")
    # model.active_adapters = adapter_name

    model_path = "./src/checkpoints/roberta_squad"
    model = RobertaForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    data = dataloader.PreprocessData(
        "squad_adversarial", "AddSent", save_data=False, save_path="../../../../"
    )
    # data = dataloader.get_dev_examples("./src/calibration/data", "dev_trivia.json")
    outputs = list()

    # def remove_white_space(example):
    #     example["question_text"] = ' '.join(example["question_text"].split())
    #     example["context_text"] = ' '.join(example["context_text"].split())
    #     return example

    grads = RandomAttributions(model=model, tokenizer=tokenizer)
    c = 0
    processed_instances = OrderedDict()
    for ex in tqdm(data.processed_val_set()):
        # ex = remove_white_space(ex)
        # print(ex)
        # ex["context"] = "Life's good when we are stress free."
        try:
            # if ex["id"] == "56f879bdaef23719006260e2":
            scores = grads.interpret([[ex["question"], ex["context"]]])
            processed_instances[ex["id"]] = scores
            c += 1
        except Exception:
            print(ex)
            print(f"Unable to get attributions: {traceback.format_exc()}")

        # else:
        #     continue
        # if c == 1:
        # break
    # import ast
    # print(ast.literal_eval(processed_instances["56f879bdaef23719006260e2"]))
    utils.dump_to_bin(
        processed_instances, "./src/results/squad_adv_random_roberta_base_info.bin"
    )
    print(f"Saved instances: {c}")
