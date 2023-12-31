import numpy as np
import torch
from transformers import (
    RealmConfig,
    RealmReader,
    RealmRetriever,
    RealmScorer,
    RealmForOpenQA,
    RealmTokenizerFast,
    load_tf_weights_in_realm,
)
from transformers.models.realm.retrieval_realm import convert_tfrecord_to_np


def add_additional_documents(openqa, additional_documents_path):
    documents = np.load(additional_documents_path, allow_pickle=True)
    total_documents = documents.shape[0]

    retriever = openqa.retriever
    tokenizer = openqa.retriever.tokenizer

    # docs
    retriever.block_records = np.concatenate(
        (retriever.block_records, documents), axis=0
    )

    # embeds
    documents = [doc.decode() for doc in documents]
    inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        projected_score = openqa.embedder(**inputs, return_dict=True).projected_score
        openqa.block_emb = torch.cat((openqa.block_emb, projected_score), dim=0)

    openqa.config.num_block_records += total_documents


def get_openqa_tf_finetuned(args, config=None):
    if config is None:
        config = RealmConfig(hidden_act="gelu_new")

    tokenizer = RealmTokenizerFast.from_pretrained(
        "google/realm-cc-news-pretrained-embedder", do_lower_case=True
    )

    block_records = convert_tfrecord_to_np(
        args.block_records_path, config.num_block_records
    )
    retriever = RealmRetriever(block_records, tokenizer)

    openqa = RealmForOpenQA(config, retriever)

    openqa = load_tf_weights_in_realm(openqa, config, args.checkpoint_path,)

    openqa = load_tf_weights_in_realm(openqa, config, args.block_emb_path,)

    openqa.eval()

    return openqa


def get_openqa_tf_pretrained(args, config=None):
    if config is None:
        config = RealmConfig(hidden_act="gelu_new")

    tokenizer = RealmTokenizerFast.from_pretrained(
        "google/realm-cc-news-pretrained-embedder", do_lower_case=True
    )

    block_records = convert_tfrecord_to_np(
        args.block_records_path, config.num_block_records
    )
    retriever = RealmRetriever(block_records, tokenizer)

    openqa = RealmForOpenQA(config, retriever)

    openqa = load_tf_weights_in_realm(openqa, config, args.bert_path,)

    openqa = load_tf_weights_in_realm(openqa, config, args.embedder_path,)

    openqa = load_tf_weights_in_realm(openqa, config, args.block_emb_path,)

    openqa.eval()

    return openqa


def get_openqa(args, config=None):
    if config is None:
        config = RealmConfig(hidden_act="gelu_new")

    retriever = RealmRetriever.from_pretrained(args.checkpoint_pretrained_name)

    openqa = RealmForOpenQA.from_pretrained(
        args.checkpoint_pretrained_name, retriever=retriever, config=config,
    )
    openqa.eval()

    return openqa


def get_scorer_reader_tokenizer_tf(args, config=None):
    if config is None:
        config = RealmConfig(hidden_act="gelu_new")
    scorer = RealmScorer(config, args.block_records_path)

    # Load retriever weights
    scorer = load_tf_weights_in_realm(scorer, config, args.retriever_path,)

    # Load block_emb weights
    scorer = load_tf_weights_in_realm(scorer, config, args.block_emb_path,)
    scorer.eval()

    reader = RealmReader.from_pretrained(
        args.checkpoint_path, config=config, from_tf=True,
    )
    reader.eval()

    tokenizer = RealmTokenizerFast.from_pretrained(
        "google/realm-cc-news-pretrained-embedder", do_lower_case=True
    )

    return scorer, reader, tokenizer


def get_scorer_reader_tokenizer_pt_pretrained(args, config=None):
    if config is None:
        config = RealmConfig(hidden_act="gelu_new")
    scorer = RealmScorer.from_pretrained(
        args.retriever_pretrained_name, args.block_records_path, config=config
    )

    # Load block_emb weights
    scorer = load_tf_weights_in_realm(scorer, config, args.block_emb_path,)
    scorer.eval()

    reader = RealmReader.from_pretrained(args.checkpoint_pretrained_name, config=config)
    reader.eval()

    tokenizer = RealmTokenizerFast.from_pretrained(
        "google/realm-cc-news-pretrained-embedder", do_lower_case=True
    )

    return scorer, reader, tokenizer


def get_scorer_reader_tokenizer_pt_finetuned(args, config=None):
    if config is None:
        config = RealmConfig(hidden_act="gelu_new")
    scorer = RealmScorer.from_pretrained(
        args.retriever_pretrained_name, args.block_records_path, config=config
    )
    scorer.eval()

    reader = RealmReader.from_pretrained(args.checkpoint_pretrained_name, config=config)
    reader.eval()

    tokenizer = RealmTokenizerFast.from_pretrained(
        "google/realm-cc-news-pretrained-embedder", do_lower_case=True
    )

    return scorer, reader, tokenizer


def get_scorer_reader_tokenizer(args, config=None):
    if config is None:
        config = RealmConfig(hidden_act="gelu_new")

    scorer = RealmScorer(config, args.block_records_path)
    reader = RealmReader(config)
    tokenizer = RealmTokenizerFast.from_pretrained(
        "google/realm-cc-news-pretrained-embedder", do_lower_case=True
    )

    return scorer, reader, tokenizer
