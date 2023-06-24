import os

import numpy as np
import torch
import wandb
from src.calibration.baseline.dataloader import PreprocessData

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    T5Config
)

# BASE_PATH = "/home/sachdeva/projects/exp_calibration/src/rag/"
# CACHE_DIR = "/home/sachdeva/.cache"
BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/src/rag/"
CACHE_DIR = "/storage/ukp/work/sachdeva/.cache"

os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['WANDB_PROJECT'] = "t5-squad-qg"
# wandb.init(dir=os.getenv("WANDB_DIR", BASE_PATH))

class T5ForQGeneration(T5ForConditionalGeneration):
    """T5ForConditionalGeneration for QA"""

    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs[0]

        return {"loss": loss}

class QGen:
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


    def _preprocess_data(self, example):
        question = example["question"].strip()
        context = example["context"]
        answer = example["answers"]
        answer_text = answer["text"][0]
        start_token = answer["answer_start"][0]
        end_token = start_token + len(answer_text)
        sep_token = ">>"

        input = f"{question} {sep_token} {context}"
        label = answer_text

        input_ids = self.tokenizer(input, max_length=512, truncation=True)["input_ids"]
        label_ids = self.tokenizer(label, max_length=256, padding=True, truncation=True)
        label_ids = np.array(label_ids["input_ids"])

        # print(tokenizer.pad_token_id)
        # print(label_ids)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        example["input_ids"] = input_ids
        example["labels"] = label_ids
        return example


    def _prepare_data(self):

        """
        Load model, tokenizer and data
        """

        config = T5Config.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_fast=True,
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [">>"]}
        )

        self.model = T5ForQGeneration.from_pretrained(
            self.model_name,
            config=config,
            cache_dir=self.cache_dir,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        dataloader = PreprocessData(
            self.dataset_name,
            self.dataset_config,
            cf_path=None,
            save_data=False,
            save_path=""
        )
        self.train_set, self.val_set = dataloader.processed_train_val_set()

    def collate_fn(self, features, pad_id=0, threshold=512):
        def pad_elems(ls, pad_id, maxlen):
            while len(ls) < maxlen:
                ls.append(pad_id)
            return ls

        maxlen = max([len(x["input_ids"]) for x in features])
        # avoid attention_type switching
        if maxlen < threshold:
            maxlen = threshold

        # dynamic padding
        input_ids = [pad_elems(x["input_ids"], pad_id=pad_id, maxlen=maxlen) for x in features]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # print(input_ids)
        # print(len(input_ids[0]))

        # padding mask...
        attention_mask = input_ids.clone()
        attention_mask[attention_mask != pad_id] = 1
        attention_mask[attention_mask == pad_id] = 0
        # replace padding token id's of the labels by -100, so it's ignored by the loss
        labels = [pad_elems(x["labels"], pad_id=-100, maxlen=256) for x in features]
        labels = torch.tensor(labels, dtype=torch.long)
        # print(labels)
        # print(len(labels[0]))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def train(self):

        args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=False,
            do_train=DO_TRAIN,
            do_eval=DO_EVAL,
            evaluation_strategy="epoch",
            # eval_steps=4000,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            warmup_steps=100,
            lr_scheduler_type="linear",
            num_train_epochs=MAX_EPOCHS,
            logging_steps=10,
            # save_steps=250,
            save_total_limit=2,
            save_strategy="epoch",
            load_best_model_at_end=False,
            run_name=WANDB_RUN_NAME,
            disable_tqdm=False,
            report_to=["wandb"],
            remove_unused_columns=False,
            fp16=FP16,
            seed=SEED,
            label_names=["labels"],  # it's important to log eval_loss
        )
        print("Batch Size", args.train_batch_size)
        print("Parallel Mode", args.parallel_mode)

        train_data = self.train_set.map(self._preprocess_data)
        val_data = self.val_set.map(self._preprocess_data)

        trainer = Trainer(
            model=self.model,
            args=args,
            data_collator=self.collate_fn,
            train_dataset=train_data,
            eval_dataset=val_data,
        )
        try:
            checkpoint = None
            if RESUME_TRAINING is not None:
                checkpoint = RESUME_TRAINING
            trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model(OUTPUT_DIR)
            self.tokenizer.save_pretrained(save_directory=OUTPUT_DIR)
        except KeyboardInterrupt:
            trainer.save_model("interrupted-squad-qg")
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Passing arguments for model, tokenizer, and dataset.")
    parser.add_argument(
        "--model_name",
        default="roberta-base",
        type=str, required=False, help="Specify the model to use.")
    parser.add_argument("--tokenizer", default="roberta-base", type=str, required=False,
                        help="Specify the tokenizer to use.")
    parser.add_argument("--output_dir", type=str, required=True, help="Specify the output directory.")
    parser.add_argument("--seed", default=42, type=int, required=False, help="Specify the seed for reproducibility.")
    parser.add_argument("--cf_path", default=None, type=str, required=False, help="Specify the path to counterfactuals.")
    parser.add_argument("--epochs", default=3, type=int, required=False, help="Epochs to train.")
    parser.add_argument("--batch_size", default=4, type=int, required=False, help="Train batch size.")
    parser.add_argument("--lr", default=2.0e-5, type=float, required=False, help="Learning rate.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float, required=False, help="Warmup ratio.")

    args = parser.parse_args()

    # hyperparameters
    LEARNING_RATE = args.lr
    WARMUP_RATIO = args.warmup_ratio
    MAX_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    OUTPUT_DIR = args.output_dir
    SEED = args.seed

    CHECKPOINT = None
    RESUME_TRAINING = None

    DO_TRAIN = True
    DO_EVAL = True
    FP16 = False

    # set to None when using all data
    MAX_TRAIN_SAMPLES = None
    MAX_VAL_SAMPLES = None

    WANDB_RUN_NAME = OUTPUT_DIR

    wandb.init(config={
        "lr": LEARNING_RATE, "warmup_ratio": WARMUP_RATIO,
        "epochs": MAX_EPOCHS,
        "model": args.model_name, "batch_size": BATCH_SIZE, "output_dir": OUTPUT_DIR,
    })
    wandb.run.name = WANDB_RUN_NAME

    trainer = QGen(
        dataset_name="squad",
        dataset_config="plain_text",
        cf_path=args.cf_path,
        # "src/data/squad/counterfactual_data_gpt_neox_20b_v2_qg_pipeline_all_data_cleaned.jsonl",
        max_src_len=384,
        stride=128,
        do_train=DO_TRAIN,
        do_eval=DO_EVAL,
        model_name=args.model_name,
        max_train_samples=MAX_TRAIN_SAMPLES,
        max_val_samples=MAX_VAL_SAMPLES,
        save_results=True
    )
    trainer.train()
