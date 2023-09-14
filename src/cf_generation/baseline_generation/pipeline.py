import os
import torch
import jsonlines
from tqdm import tqdm
import numpy as np
from typing import List
from collections import OrderedDict
from transformers import (
    RealmRetriever,
    RealmTokenizer,
    RealmConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from src.cf_generation.baseline_generation.modelling_realm import RealmForOpenQA
from src.calibration.baseline import dataloader

BASE_PATH = os.getenv("PYTHONPATH", "/home/sachdeva/projects/exp_calibration/")
NUM_BEAMS = 15


class CounterfactualGeneration:
    def __init__(self, beam_size: int = 5, prefix: str = None):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hl_token = "<hl>"
        self.prefix = prefix

        # --------------------------------
        # load retriever model
        # --------------------------------
        self.retriever = RealmRetriever.from_pretrained("google/realm-orqa-nq-openqa")
        self.realm_tokenizer = RealmTokenizer.from_pretrained(
            "google/realm-orqa-nq-openqa"
        )
        self.realm_tokenizer.add_tokens(self.hl_token)

        self.hl_token_id = self.realm_tokenizer.convert_tokens_to_ids(["<hl>"])[0]

        realm_config = RealmConfig.from_pretrained("google/realm-orqa-nq-openqa")
        # print(config)
        realm_config.reader_beam_size = beam_size
        self.realm_model = RealmForOpenQA.from_pretrained(
            "google/realm-orqa-nq-openqa",
            retriever=self.retriever,
            config=realm_config,
            torch_dtype=torch.bfloat16,
        )
        self.realm_model.to(self.device)

        # --------------------------------
        # load question generation model
        # --------------------------------
        model_path = BASE_PATH + "t5-3b-squad-qg-seed-42"
        self.qg_tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.qg_tokenizer.add_special_tokens(
        #     {"additional_special_tokens": [self.hl_token]}
        # )
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        self.qg_model.to(self.device)
        self.qg_model.eval()

    def _retrieve_context_and_answers(self, question: str = None):
        question_ids = self.realm_tokenizer([question], return_tensors="pt").to(
            self.device
        )
        reader_output = self.realm_model(**question_ids, return_dict=True)
        start_positions = reader_output.reader_output.start_pos.cpu().numpy()
        end_positions = reader_output.reader_output.end_pos.cpu().numpy()
        # add hl_token_id at tensor index start position and end position
        highlight_input_ids = reader_output.input_ids.cpu().numpy()
        inputs: List = []
        for idx, (start, end) in enumerate(zip(start_positions, end_positions)):
            start = int(start)
            end = int(end)
            # get sep idx to retrieve context
            sep_idx = np.where(
                highlight_input_ids[idx, :start] == self.realm_tokenizer.sep_token_id
            )[0][-1]
            inputs.append(
                np.concatenate(
                    [
                        highlight_input_ids[idx, sep_idx:start],
                        [self.hl_token_id],
                        highlight_input_ids[idx, start : end + 1],
                        [self.hl_token_id],
                        highlight_input_ids[idx, end + 1 :],
                    ],
                    axis=0,
                )
            )

        predicted_answers = self.realm_tokenizer.batch_decode(
            reader_output.predicted_answer_ids, skip_special_tokens=True
        )
        contexts = self.realm_tokenizer.batch_decode(
            torch.tensor(np.array(inputs), device=self.device),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        retrieved_blocks = np.take(
            self.retriever.block_records,
            indices=reader_output.retrieved_block_ids.detach().cpu().numpy(),
            axis=0,
        )

        return (
            contexts,
            predicted_answers,
            retrieved_blocks,
            start_positions,
            end_positions,
        )

    def generate_question(self, question: str = None):
        (
            contexts,
            predicted_answers,
            retrieved_blocks,
            start_pos,
            end_pos,
        ) = self._retrieve_context_and_answers(question)
        prepared_inputs = [self.prefix + context for context in contexts]
        features = self.qg_tokenizer(
            prepared_inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        # print(features)
        outputs = self.qg_model.generate(
            **features, max_length=128, num_beams=NUM_BEAMS, early_stopping=True
        )
        predicted_questions = self.qg_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        return (
            predicted_questions,
            predicted_answers,
            retrieved_blocks,
            start_pos,
            end_pos,
        )


def save_to_disk(data, file_name):
    with jsonlines.open(file_name, "a") as writer:
        for example in tqdm(data, total=len(data), desc="Saving samples ... "):
            # print(example)
            for idx, (question, context, answer, start_pos, end_pos) in enumerate(
                zip(
                    example["predicted_questions"],
                    example["retrieved_contexts"],
                    example["predicted_answers"],
                    example["start_positions"],
                    example["end_positions"],
                )
            ):
                # print(question)
                # print(question, context, answer, start_pos, end_pos)
                writer.write(
                    {
                        "id": example["id"] + "_" + str(idx),
                        "question": example["question"],
                        "context": example["context"],
                        "answers": example["answers"],
                        "predicted_question": question,
                        "retrieved_context": context.decode("utf-8"),
                        # "original_retrieved_context": context,
                        "predicted_answer": {
                            "text": answer,
                            "answer_start": int(start_pos),
                            "answer_end": int(end_pos),
                        },
                    },
                )


if __name__ == "__main__":
    data = dataloader.PreprocessData(
        "squad", "plain_text", save_data=False, save_path="../../../"
    )

    prefix = "generate question: "
    cg = CounterfactualGeneration(beam_size=15, prefix=prefix)

    c = 0
    cache_file_name = (
        BASE_PATH
        + "src/data/squad/rag_counterfactuals_with_answers_t5_3b_squad_qg_06082023"
    )
    processed_instances = OrderedDict()
    outputs = []
    train_data, val_data = data.processed_train_val_set()
    chunk_size = 1000
    train_len = len(train_data)
    num_chunks = int(train_len / chunk_size) + 1
    for ex in tqdm(train_data):
        # ex = remove_white_space(ex)
        try:
            c += 1
            # if c<=87000:
            #     continue

            (
                predicted_questions,
                predicted_answers,
                retrieved_blocks,
                start_pos,
                end_pos,
            ) = cg.generate_question(ex["question"])
            # processed_instances[ex["id"]] = data
            outputs.append(
                {
                    "id": ex["id"],
                    "question": ex["question"],
                    "context": ex["context"],
                    "answers": ex["answers"],
                    "predicted_questions": predicted_questions,
                    "retrieved_contexts": retrieved_blocks,
                    "predicted_answers": predicted_answers,
                    "start_positions": list(start_pos),
                    "end_positions": list(end_pos),
                }
            )

            if c % 1000 == 0:
                save_to_disk(outputs, file_name=cache_file_name + ".jsonl")
                outputs = []
                num_chunks -= 1
            # save last chunk
            # if num_chunks == 1 and c == train_len:
            #     save_to_disk(outputs, file_name=cache_file_name + ".jsonl")
            #     outputs = []

        except Exception as e:
            print(f"Unable to get counterfactuals: {e}")
            print(ex)

    save_to_disk(outputs, file_name=cache_file_name + ".jsonl")
    # if c == 1:
    #     break
    # print(processed_instances)

    # data = dataloader.get_dev_examples(BASE_PATH + "src/data", "dev_hotpot.json")
    # data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/SearchQA-dev.jsonl")
    #
    # def remove_white_space(example):
    #     example["question_text"] = ' '.join(example["question_text"].split())
    #     example["context_text"] = ' '.join(example["context_text"].split())
    #     return example
    #
    # c = 0
    # cache_file_name = BASE_PATH + "src/data/search_qa/rag_counterfactuals"
    # processed_instances = OrderedDict()
    # outputs = []
    # for ex in tqdm(data):
    #     ex = remove_white_space(ex)
    #     try:
    #         # if ex["id"] == "56e1239acd28a01900c67641":
    #         #     print(ex)
    #         predicted_questions, predicted_answers, retrieved_blocks = cg.generate_question(ex["question_text"])
    #         # processed_instances[ex["id"]] = data
    #         outputs.append({
    #             "id": ex["qas_id"],
    #             "question": ex["question_text"],
    #             "context": ex["context_text"],
    #             "answers": ex["answer_text"],
    #             "predicted_questions": predicted_questions,
    #             "retrieved_contexts": retrieved_blocks,
    #             "predicted_answers": predicted_answers,
    #         })
    #         # print(tag_info)
    #         c += 1
    #         if c % 1000 == 0:
    #             save_to_disk(outputs, file_name=cache_file_name + ".jsonl")
    #             outputs = []
    #     except Exception as e:
    #         print(f"Unable to get counterfactuals: {e}")
    #         print(ex)
