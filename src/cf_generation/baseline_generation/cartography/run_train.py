"""
Train roberta model with counterfactuals
"""

import os
import logging
import json
from typing import List, Tuple
import collections
import pandas as pd
import numpy as np

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)

import wandb
from src.cartography.dataloader import PreprocessData

logger = logging.getLogger(__name__)

# hyperparameters
LEARNING_RATE = 1.0e-5
WARMUP_RATIO = 0.06
MAX_EPOCHS = 5
FP16 = False
MODEL_NAME = "roberta-base"
BATCH_SIZE = 16

RESUME_TRAINING = None

OUTPUT_DIR = "roberta-squad-t5-squad-cfs-cartography"
CHECKPOINT = None

DO_TRAIN = True
DO_EVAL = True


# set to None when using all data
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

WANDB_RUN_NAME = OUTPUT_DIR

wandb.init(config={
    "lr": LEARNING_RATE, "warmup_ratio": WARMUP_RATIO,
    "epochs": MAX_EPOCHS,
    "model": MODEL_NAME, "batch_size": BATCH_SIZE, "output_dir": OUTPUT_DIR,
})
wandb.run.name = WANDB_RUN_NAME
BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"


class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ep_train_ids = None
        self.ep_start_logits = None
        self.ep_end_logits = None
        self.ep_gold_start = None
        self.ep_gold_end = None

    def training_step(self, model: torch.nn.Module, inputs: dict) -> torch.Tensor:
        model.train()
        idx = inputs.pop("idx")

        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        loss = outputs[0]

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        epoch = int(self.state.epoch)
        start_positions = inputs["start_positions"].detach().cpu().numpy()
        end_positions = inputs["end_positions"].detach().cpu().numpy()
        start_logits = outputs[1].detach().cpu().numpy()
        end_logits = outputs[2].detach().cpu().numpy()

        # store the logits
        if self.ep_train_ids is None:
            self.ep_train_ids = np.array(idx)
            self.ep_start_logits = start_logits
            self.ep_end_logits = end_logits
            self.ep_gold_start = start_positions
            self.ep_gold_end = end_positions
        else:
            self.ep_train_ids = np.concatenate((self.ep_train_ids,  np.array(idx)), axis=0)
            self.ep_start_logits = np.concatenate((self.ep_start_logits, start_logits), axis=0)
            self.ep_end_logits = np.concatenate((self.ep_end_logits, end_logits), axis=0)
            self.ep_gold_start = np.concatenate((self.ep_gold_start, start_positions), axis=0)
            self.ep_gold_end = np.concatenate((self.ep_gold_end, end_positions), axis=0)

        # log the metrics after every epoch
        if len(self.ep_train_ids) == len(self.train_dataset):
            self.log_training_dynamics(
                self.args.output_dir,
                epoch,
                list(self.ep_train_ids),
                list(self.ep_start_logits),
                list(self.ep_end_logits),
                list(self.ep_gold_start),
                list(self.ep_gold_end),
            )
            self.ep_train_ids = None
            self.ep_start_logits = None
            self.ep_end_logits = None
            self.ep_gold_start = None
            self.ep_gold_end = None

        return loss.detach()

    @staticmethod
    def log_training_dynamics(
            output_dir: os.path,
            epoch: int,
            train_ids: List[int],
            start_logits: List[List[float]],
            end_logits: List[List[float]],
            start_golds: List[int],
            end_golds: List[int]
        ):
        """
        Save training dynamics (logits) from given epoch as records of a `.jsonl` file.
        """
        td_df = pd.DataFrame(
            {
                "guid": train_ids,
                f"start_logits_epoch_{epoch}": start_logits,
                f"end_logits_epoch_{epoch}": end_logits,
                "start_gold": start_golds,
                "end_gold": end_golds
            }
        )

        logging_dir = os.path.join(output_dir, f"training_dynamics")
        # Create directory for logging training dynamics, if it doesn't already exist.
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)
        epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{epoch}.jsonl")
        td_df.to_json(epoch_file_name, lines=True, orient="records")
        logger.info(f"Training Dynamics logged to {epoch_file_name}")


class RobertaSquad:
    def __init__(
            self,
            dataset_name: str,
            dataset_config: str,
            cf_path: str,
            max_src_len: int,
            stride: int,
            do_train: bool,
            do_eval: bool,
            model_name: str,
            cache_dir: str=None,
            max_train_samples: int=None,
            max_val_samples: int=None,
            save_results: bool=False
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.cf_path = cf_path
        self.max_src_len = max_src_len
        self.stride = stride
        self.do_train = do_train
        self.do_eval = do_eval
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.padding = "max_length"
        self.save_results = save_results
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load resources
        self._prepare_data()

    def _ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.
        Args:
            inputs (keyword arguments that should be :obj:`torch.Tensor`):
                The tensors to place on :obj:`self.device`.
        Return:
            :obj:`Dict[str, torch.Tensor]`: The same as :obj:`inputs` but on the proper device.
        """
        return {name: tensor.to(self.model.device) for name, tensor in inputs.items()}

    def _prepare_data(self):

        """
        Load model, tokenizer and data
        """

        config = AutoConfig.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_fast=True,
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_name,
            config=config,
            cache_dir=self.cache_dir,
        )
        dataloader = PreprocessData(
            self.dataset_name,
            self.dataset_config,
            cf_path=BASE_PATH + self.cf_path,
            save_data=False,
            save_path=""
        )
        if self.do_train and not self.do_eval:
            self.train_set, _ = dataloader.processed_counterfactuals()
        elif self.do_train and self.do_eval:
            self.train_set, self.val_set = dataloader.processed_counterfactuals()

    def _preprocess_training_examples(self, examples):
        """
        Preprocess training examples
        """

        inputs = self.tokenizer(
            examples["question"],
            examples["context"],
            max_length=self.max_src_len,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        batch_ids = examples["id"]

        processed_batch_ids = []
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            processed_batch_ids.append(batch_ids[sample_idx])
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        inputs["id"] = processed_batch_ids

        return inputs

    def data_collator(self, features):
        idx = [x["id"] for x in features]
        input_ids = [x["input_ids"] for x in features]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = [x["attention_mask"] for x in features]
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        start_positions = [x["start_positions"] for x in features]
        start_positions = torch.tensor(start_positions, dtype=torch.long)
        end_positions = [x["end_positions"] for x in features]
        end_positions = torch.tensor(end_positions, dtype=torch.long)

        return {
            "idx": idx,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": start_positions,
            "end_positions": end_positions,
        }

    def train(self):
        if self.do_train:
            logger.info("*** Train ***")

            if self.max_train_samples is not None:
                # We will select sample from whole data if argument is specified
                max_train_samples = min(len(self.train_set), self.max_train_samples)
                self.train_set = self.train_set.select(range(max_train_samples))

            train_dataset = self.train_set.map(
                self._preprocess_training_examples,
                batched=True,
                remove_columns=self.train_set.column_names,
            )
            print(len(self.train_set), len(train_dataset))

        if self.do_eval:
            if self.max_val_samples is not None:
                # We will select sample from whole data if argument is specified
                max_val_samples = min(len(self.val_set), self.max_val_samples)
                self.val_set = self.val_set.select(range(max_val_samples))

            validation_dataset = self.val_set.map(
                self._preprocess_training_examples,
                batched=True,
                remove_columns=self.val_set.column_names,
            )
            print(len(self.val_set), len(validation_dataset))

        args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=False,
            do_train=self.do_train,
            do_eval=self.do_eval,
            # evaluation_strategy="epoch",
            save_strategy="epoch",
            # eval_steps=4000,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            num_train_epochs=MAX_EPOCHS,
            logging_steps=10,
            save_total_limit=2,
            # load_best_model_at_end=True,
            run_name=WANDB_RUN_NAME,
            disable_tqdm=False,
            report_to=["wandb"],
            remove_unused_columns=False,
            fp16=FP16,
            # label_names=[
            #     "eval_loss",
            # ],  # it's important to log eval_loss
        )
        print("Batch Size", args.train_batch_size)
        print("Parallel Mode", args.parallel_mode)

        trainer = CustomTrainer(
            model=self.model,
            args=args,
            data_collator=self.data_collator,
            train_dataset=train_dataset if self.do_train else None,
            eval_dataset=validation_dataset if self.do_eval else None,
        )
        try:
            if self.do_train:
                checkpoint = None
                if RESUME_TRAINING is not None:
                    checkpoint = RESUME_TRAINING
                trainer.train(resume_from_checkpoint=checkpoint)
                trainer.save_model()
        except KeyboardInterrupt:
            trainer.save_model("interrupted-squad")

        wandb.finish()


if __name__ == '__main__':
    trainer = RobertaSquad(
        dataset_name="squad",
        dataset_config="plain_text",
        cf_path="src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_min_filtered_final_dedup_1.jsonl",
        max_src_len=384,
        stride=128,
        do_train=DO_TRAIN,
        do_eval=DO_EVAL,
        model_name=MODEL_NAME,
        max_train_samples=MAX_TRAIN_SAMPLES,
        max_val_samples=MAX_VAL_SAMPLES,
        save_results=True
    )
    trainer.train()
