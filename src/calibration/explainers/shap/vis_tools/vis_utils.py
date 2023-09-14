# from PIL import ImageFont, ImageDraw, Image
# from colour import Color
import numpy as np
import math
from itertools import chain
from typing import List

import os
from os.path import join
import shutil
import string
from transformers import RobertaTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from .vis_attention import visualize_connection, merge_attention_by_segments
from .vis_token import visualize_tok_attribution, merge_token_attribution_by_segments
from .vis_vanilla_token import visualize_vanilla_tok_attribution

# from src.rag.shap.data.dataset_utils import merge_tokens_into_words


def _mkdir_f(prefix):
    if os.path.exists(prefix):
        shutil.rmtree(prefix)
    os.makedirs(prefix)


def visualize_token_attributions(args, tokenizer, interp_info, fname):
    assert args.visual_dir is not None
    # prefix
    prefix = join(args.visual_dir, fname.split(".")[0])
    # attribution
    # N Layer * N Head
    attribution = interp_info["attribution"]
    attribution_val = attribution.numpy()
    tokens = list(interp_info["feature"][0])
    words, segments = _bpe_decode(tokenizer, tokens)

    # plot aggregated
    # along layers
    aggregated_attribution = merge_token_attribution_by_segments(
        attribution_val, segments
    )
    visualize_vanilla_tok_attribution(
        prefix + ".jpg", words, aggregated_attribution, interp_info
    )


def _bpe_decode(
    tokenizer,
    tokens: List[str],
    # attributions: List
):

    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    decoded_each_tok = [
        bytearray([byte_decoder[c] for c in t]).decode(
            encoding="utf-8", errors="replace"
        )
        for t in tokens
    ]

    end_points = []
    force_break = False
    for idx, token in enumerate(decoded_each_tok):
        # special token, punctuation, alphanumeric
        if (
            token in tokenizer.all_special_tokens
            or token in string.punctuation
            or not any([x.isalnum() for x in token.lstrip()])
            or token.lstrip == "'s"
        ):
            end_points.append(idx)
            force_break = True
            continue

        if force_break:
            end_points.append(idx)
            force_break = False
            continue

        if token[0] == " ":
            tokens[idx] = token[:]
            end_points.append(idx)

    end_points.append(len(tokens))
    segments = []
    for i in range(1, len(end_points)):
        if end_points[i - 1] == end_points[i]:
            continue
        segments.append((end_points[i - 1], end_points[i]))

    merged_tokens = []
    for s0, s1 in segments:
        merged_tokens.append("".join(decoded_each_tok[s0:s1]))

    return merged_tokens, segments


def visualize_attributions(args, tokenizer, interp_info, do_head=False, do_layer=True):
    assert args.visual_dir is not None
    feature = interp_info["feature"]

    # prefix
    prefix = join(args.visual_dir, f"{feature.example_index}-{feature.qas_id}")
    _mkdir_f(prefix)

    # attribution
    # N Layer * N Head
    attribution = interp_info["attribution"]
    # attribution = attribution.res
    n_layers, n_heads, n_tokens, _ = tuple(attribution.size())

    attribution_val = attribution.numpy()
    attribution_diff = np.sum(attribution_val)
    attribution_val = attribution_val / attribution_diff

    words, segments = merge_tokens_into_words(tokenizer, interp_info["feature"])

    # plot aggregated
    # along layers
    aggregated_attribution = np.sum(attribution_val, axis=0)
    aggregated_attribution = np.sum(aggregated_attribution, axis=0)
    aggregated_attribution = merge_attention_by_segments(
        aggregated_attribution, segments
    )
    visualize_connection(
        join(prefix, "aggregated.jpg"), words, aggregated_attribution, interp_info
    )
    visualize_tok_attribution(
        join(prefix, "token_attribution.jpg"),
        words,
        aggregated_attribution,
        interp_info,
    )
    if do_head:
        aggregated_by_head = np.sum(attribution_val, axis=0)
        for i_head in range(aggregated_by_head.shape[0]):
            aggregated_head_i = aggregated_by_head[i_head]
            aggregated_head_i = merge_attention_by_segments(aggregated_head_i, segments)
            visualize_connection(
                join(prefix, f"head-{i_head}.jpg"),
                words,
                aggregated_head_i,
                interp_info,
            )
    if do_layer:
        aggregated_by_layer = np.sum(attribution_val, axis=1)
        for i_layer in range(aggregated_by_layer.shape[0]):
            aggregated_layer_i = aggregated_by_layer[i_layer]
            aggregated_layer_i = merge_attention_by_segments(
                aggregated_layer_i, segments
            )
            visualize_connection(
                join(prefix, f"layer-{i_layer}.jpg"),
                words,
                aggregated_layer_i,
                interp_info,
            )


def visualize_layer_attributions(args, tokenizer, interp_info, do_layer=True):
    assert args.visual_dir is not None
    feature = interp_info["feature"]

    # prefix
    prefix = join(args.visual_dir, f"{feature.example_index}-{feature.qas_id}")
    _mkdir_f(prefix)

    # attribution
    # N Layer * N Head
    attribution = interp_info["attribution"]
    attention = interp_info["attention"]
    prelim_result = interp_info["prelim_result"]
    # attribution = attribution.res
    n_layers, n_heads, n_tokens, _ = tuple(attribution.size())

    attribution_val = attribution.numpy()

    active_layers = interp_info["active_layers"]
    words, segments = merge_tokens_into_words(tokenizer, interp_info["feature"])

    aggregated_attribution = np.sum(attribution_val, axis=0)
    aggregated_attribution = np.sum(aggregated_attribution, axis=0)
    aggregated_attribution = merge_attention_by_segments(
        aggregated_attribution, segments
    )
    visualize_connection(
        join(prefix, "aggregated.jpg"), words, aggregated_attribution, interp_info
    )
    visualize_tok_attribution(
        join(prefix, "token_attribution.jpg"),
        words,
        aggregated_attribution,
        interp_info,
    )

    if do_layer:
        aggregated_by_layer = np.sum(attribution_val, axis=1)
        for i_layer in range(aggregated_by_layer.shape[0]):
            if not active_layers[i_layer]:
                continue
            aggregated_layer_i = aggregated_by_layer[i_layer]
            aggregated_layer_i = merge_attention_by_segments(
                aggregated_layer_i, segments
            )
            visualize_connection(
                join(prefix, f"layer-{i_layer}.jpg"),
                words,
                aggregated_layer_i,
                interp_info,
            )


def visualize_pruned_layer_attributions(args, tokenizer, interp_info, do_layer=True):
    assert args.visual_dir is not None
    feature = interp_info["feature"]

    # prefix
    prefix = join(args.visual_dir, f"{feature.example_index}-{feature.qas_id}")
    _mkdir_f(prefix)

    # attribution
    # N Layer * N Head
    attribution = interp_info["attribution"]
    attention = interp_info["attention"]
    prelim_result = interp_info["prelim_result"]
    # attribution = attribution.res
    n_layers, n_heads, n_tokens, _ = tuple(attribution.size())

    attribution_val = attribution.numpy()

    active_layers = interp_info["active_layers"]
    words, segments = merge_tokens_into_words(tokenizer, interp_info["feature"])
    if do_layer:
        aggregated_by_layer = np.sum(attribution_val, axis=1)
        for i_layer in range(aggregated_by_layer.shape[0]):
            if not active_layers[i_layer]:
                continue
            aggregated_layer_i = aggregated_by_layer[i_layer]
            aggregated_layer_i = merge_attention_by_segments(
                aggregated_layer_i, segments
            )
            visualize_connection(
                join(prefix, f"layer-{i_layer}.jpg"),
                words,
                aggregated_layer_i,
                interp_info,
                vis_negative=False,
            )
