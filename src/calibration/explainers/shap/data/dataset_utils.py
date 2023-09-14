from transformers import RobertaTokenizer
import string
from typing import List
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


def get_prefix_tokens(dataset, tokenizer):
    if dataset == "hpqa":
        return ["yes", "no", "unk", tokenizer.sep_token]
    elif dataset in ["squad", "bioasq", "newsqa", "natq", "trivia", "hotpot"]:
        return []
    elif dataset == "synth":
        return []
    elif dataset == "simple":
        return []
    elif dataset == "comp":
        return []
    else:
        raise RuntimeError("invalid dataset")


def _merge_roberta_tokens_into_words(tokenizer, feature):
    tokens = feature[0]

    decoded_each_tok = [
        bytearray([tokenizer.byte_decoder[c] for c in t]).decode(
            "utf-8", errors=tokenizer.errors
        )
        for t in tokens
    ]

    token_to_orig_map = feature.token_to_orig_map

    end_points = []
    context_start = tokens.index(tokenizer.eos_token)
    force_break = False
    for i, t in enumerate(decoded_each_tok):
        # special token
        if t in tokenizer.all_special_tokens:
            end_points.append(i)
            force_break = True
            continue

        if t in string.punctuation:
            end_points.append(i)
            force_break = True
            continue

        if force_break:
            end_points.append(i)
            force_break = False
            continue

        # if in question segment
        if i <= context_start:
            if t[0] == " ":
                decoded_each_tok[i] = t[1:]
                end_points.append(i)
        else:
            if token_to_orig_map[i] != token_to_orig_map[i - 1]:
                end_points.append(i)
    end_points.append(len(decoded_each_tok))

    # if in context segment
    segments = []
    for i in range(1, len(end_points)):
        if end_points[i - 1] == end_points[i]:
            continue
        segments.append((end_points[i - 1], end_points[i]))

    merged_tokens = []
    for s0, s1 in segments:
        merged_tokens.append("".join(decoded_each_tok[s0:s1]))

    return merged_tokens, segments


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


def merge_tokens_into_words(tokenizer, feature):
    if isinstance(tokenizer, RobertaTokenizer):
        return _bpe_decode(tokenizer, feature)
