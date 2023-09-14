import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

# BASE_PATH = "/home/sachdeva/projects/exp_calibration/src/rag/"
# CACHE_DIR = "/home/sachdeva/.cache"
BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/src/rag/"
CACHE_DIR = "/storage/ukp/work/sachdeva/.cache"

OUTPUT_DIR = "t5-3b-nq-short-qg-seed-42"

TRAIN_ON_SMALL = os.environ.pop("TRAIN_ON_SMALL", "false")
RESUME_TRAINING = None

SEED = 42

LEARNING_RATE = 2 * 1.0e-5
WARMUP_STEPS = 100
DROPOUT = 0.1
MAX_EPOCHS = 3
FP16 = False
SCHEDULER = "linear"
MODEL_NAME = "t5-3b"

BATCH_SIZE = 4

DO_TRAIN = True
DO_EVAL = True
DO_PREDICT = True

os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["WANDB_PROJECT"] = "t5-natural-questions-qg"
# wandb.init(dir=os.getenv("WANDB_DIR", BASE_PATH))

WANDB_RUN_NAME = OUTPUT_DIR

wandb.init(
    config={
        "seed": SEED,
        "lr": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "dropout": 0.1,
        "epochs": MAX_EPOCHS,
        "scheduler": SCHEDULER,
        "model": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "output_dir": OUTPUT_DIR,
    }
)
wandb.run.name = WANDB_RUN_NAME


def collate_fn(features, pad_id=0, threshold=640):
    def pad_elems(ls, pad_id, maxlen):
        while len(ls) < maxlen:
            ls.append(pad_id)
        return ls

    maxlen = max([len(x["input_ids"]) for x in features])
    # avoid attention_type switching
    if maxlen < threshold:
        maxlen = threshold

    # dynamic padding
    input_ids = [pad_elems(x["input_ids"], pad_id=pad_id, maxlen=640) for x in features]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    # print(input_ids)
    # print(len(input_ids[0]))

    # padding mask...
    attention_mask = input_ids.clone()
    attention_mask[attention_mask != pad_id] = 1
    attention_mask[attention_mask == pad_id] = 0
    # print(attention_mask)
    # print(len(attention_mask[0]))
    # print(torch.tensor(
    #         [x["labels"] for x in features], dtype=torch.long
    #     ))
    # replace padding token id's of the labels by -100, so it's ignored by the loss
    labels = [pad_elems(x["labels"], pad_id=-100, maxlen=256) for x in features]
    labels = torch.tensor(labels, dtype=torch.long)
    # print(labels)
    # print(len(labels[0]))

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class T5ForNaturalQuestions(T5ForConditionalGeneration):
    """T5ForConditionalGeneration for NQ QA"""

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self, input_ids=None, attention_mask=None, labels=None,
    ):
        outputs = super().forward(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs[0]

        return {"loss": loss}


if __name__ == "__main__":
    # "nq-training.jsonl" & "nq-validation.jsonl" are obtained from running `prepare_nq.py`
    tr_dataset = load_dataset(
        "json",
        data_files=BASE_PATH + "data/nq-train-tokenized-short-qg.jsonl",
        cache_dir=CACHE_DIR,
    )["train"]
    val_dataset = load_dataset(
        "json",
        data_files=BASE_PATH + "data/nq-dev-tokenized-short-qg.jsonl",
        cache_dir=CACHE_DIR,
    )["train"]

    if TRAIN_ON_SMALL == "true":
        np.random.seed(SEED)
        indices = np.random.randint(0, 87000, size=100)
        tr_dataset = tr_dataset.select(indices)
        np.random.seed(SEED)
        indices = np.random.randint(0, 2000, size=5)
        val_dataset = val_dataset.select(indices)
    # print(tr_dataset, val_dataset)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<hl>"]})
    model = T5ForNaturalQuestions.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, torch_dtype=torch.float16, device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))

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
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=SCHEDULER,
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

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        train_dataset=tr_dataset,
        eval_dataset=val_dataset,
    )
    try:
        checkpoint = None
        if RESUME_TRAINING is not None:
            checkpoint = RESUME_TRAINING
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(save_directory=OUTPUT_DIR)
    except KeyboardInterrupt:
        trainer.save_model("interrupted-natural-questions")
    wandb.finish()
