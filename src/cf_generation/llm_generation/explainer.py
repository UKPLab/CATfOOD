import inseq
import torch
import json
import os
from collections import Counter
import pandas as pd
from tqdm import tqdm
from itertools import islice
import collections

# "{q} The possibile answers are {options_list with a, b, c}, but the correct of a, b, c is"

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"


def process_data():
    # read json file
    with open(f"{BASE_PATH}src/few_shot/big_bench_tasks/social_iqa.json", "r") as file:
        data = json.load(file)
        examples = data["examples"]
    questions = [sample["input"] for sample in examples]
    options = [list(sample["target_scores"].keys()) for sample in examples]
    answers = [list(sample["target_scores"].values()) for sample in examples]
    return questions, options, answers


def _merge_attributions(attributions):
    pass


def evaluate(file_name):
    correct = 0
    mapper = {"a": 0, "b": 1, "c": 2}

    with open(file_name, "r") as file:
        data = json.load(file)
    total_samples = len(data)
    for sample in tqdm(data):
        # print(sample)
        answer = sample["answer"][0]
        pred = sample["prediction"][0].strip()
        if len(pred) > 1:
            pred = pred[0]
        answer_key = mapper.get(pred, None)
        # print("answer key: ", answer_key)
        if answer_key is not None:
            if sample["options"][answer_key] == answer:
                correct += 1
        else:
            # print(sample["options"][answer_key])
            print(answer)
            print("-")
        # break

    accuracy = (correct / total_samples) * 100
    print(accuracy)


def get_attribution(model_name, attribution_type, max_samples=None, top_k=5):

    questions, options, answers = process_data()

    if max_samples:
        questions = questions[:max_samples]
        options = options[:max_samples]
        answers = answers[:max_samples]

    model = inseq.load_model(
        model_name, attribution_type, torch_dtype=torch.bfloat16, device_map="auto"
    )
    # load_in_8bit=True)

    feature_attributions = []
    topk_attributions = []
    outputs = []
    for ques, option, ans in tqdm(zip(questions, options, answers)):

        len_options = len(option)
        if len_options == 3:
            # prompt = f"Question: {ques} " \
            #          f"\n Options: a) {option[0]}, b) {option[1]}, c) {option[2]}" \
            #          "\n Answer: "
            prompt = f"{ques} The possible answers are a) {option[0]}, b) {option[1]}, c) {option[2]}, but the correct of a, b, c is"
        elif len_options == 2:
            # prompt = f"Question: {ques} " \
            #          f"\n Options: a) {option[0]}, b) {option[1]}" \
            #          "\n Answer: "
            prompt = f"{ques} The possible answers are a) {option[0]}, b) {option[1]}, but the correct of a, b is"

        attributions = model.attribute(prompt, generation_args={"max_new_tokens": 15},)
        # attributions.show()
        # print(attributions)
        # merged_attributions = inseq.FeatureAttributionOutput.merge_attributions([attributions])
        # print("Merged:", merged_attributions)
        aggr = attributions.sequence_attributions[0].aggregate()
        # print(aggr)
        # Creating a mapping of [src_token, tgt_token] -> attribution score
        # score_map = {}
        # for src_idx, src_tok in enumerate(aggr.source):
        #     for tgt_idx, tgt_tok in enumerate(aggr.target):
        #         score_map[(src_tok.token, tgt_tok.token)] = aggr.source_attributions[src_idx, tgt_idx].item()
        #
        # print(score_map)
        # print(attributions.get_scores_dicts())
        score_dict = aggr.get_scores_dicts()
        source_attributions = score_dict["source_attributions"]
        # feature_attributions.append(source_attributions)
        # df = pd.DataFrame(source_attributions)
        # print(df.head())

        averages = {}
        for token, token_dict in source_attributions.items():
            for key, value in token_dict.items():
                if key not in averages:
                    averages[key] = [value, 1]
                else:
                    averages[key][0] += value
                    averages[key][1] += 1

        averages_dict = {key: value[0] / value[1] for key, value in averages.items()}
        # print(averages_dict)

        x_before_options = {}
        for key, value in averages_dict.items():
            if key == "▁Options":
                break
            x_before_options[key] = value
        # print(x_before_options)
        sorted_averages = dict(
            sorted(x_before_options.items(), key=lambda x: x[1], reverse=True)
        )
        # print(sorted_averages)
        top_attributions = dict(islice(sorted_averages.items(), top_k))
        # topk_attributions.append(top_attributions)

        output = collections.OrderedDict()
        output["question"] = ques
        output["options"] = option
        output["answer"] = [option[ans.index(1)]]
        output["prediction"] = attributions.info["generated_texts"]
        output["attribution"] = top_attributions
        outputs.append(output)
        # print(output)

    save_file_name = model_name.split("/")[-1] + f"_{attribution_type}_social_iqa"
    output_nbest_file = os.path.join(
        BASE_PATH, f"predictions_{save_file_name}_prompt_2.json"
    )
    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(outputs, indent=4) + "\n")

    return feature_attributions


if __name__ == "__main__":
    # source_attributions = get_attribution(
    #     model_name="google/flan-t5-large",
    #     attribution_type="attention",
    #     max_samples=None,
    #     top_k=5
    # )
    # print(source_attributions)
    # top_attrs = postprocess_attributions(source_attributions, top_k=10)
    # print(top_attrs)
    # process_data()

    ########### EVALUATE ############
    model_name = "flan-t5-large"
    evaluate(os.path.join(BASE_PATH, f"predictions_{model_name}_attention.json"))

    # x = [{'▁': {'▁Question': 0.0004824649658985436, ':': 0.09963615983724594, '▁Tracy': 0.023480601608753204,
    #             '▁didn': 0.0008805241086520255, "'": 0.0013725516619160771, 't':0.0015102395555004478,
    #             '▁go': 0.0005502908607013524, '▁home': 0.0014923501294106245, '▁that': 0.0009823078289628029,
    #             '▁evening': 0.0018404676811769605, '▁and': 0.00046898151049390435, '▁': 0.09211429953575134,
    #             're': 0.0004362099862191826, 'sisted': 0.0027353481855243444, '▁Riley': 0.007774821948260069,
    #             's': 0.0036468186881393194,'▁attacks': 0.004170892760157585, '.': 0.0030060261487960815,
    #             '▁What': 0.002392962109297514, '▁does': 0.0019515401218086481, '▁need': 0.013643212616443634,
    #             '▁to': 0.0005810720031149685, '▁do': 0.0037925769574940205, '▁before': 0.00945215579122305,
    #             '▁this': 0.0017502920236438513, '?': 0.002131211804226041, '▁Options': 0.013728966936469078,
    #             'a': 0.004350507166236639, ')': 0.0063089970499277115, '▁Make': 0.05659395083785057,
    #             '▁new': 0.0028916974551975727, '▁plan': 0.0028365186881273985, ',': 0.013565066270530224,
    #             'b': 0.008598132990300655, '▁Go': 0.0036058370023965836, '▁see': 0.0018858902622014284 }},
    # {'▁': {'▁Question': 0.0004824649658985436, ':': 0.09963615983724594, '▁Tracy': 0.023480601608753204,
    #        '▁didn': 0.0008805241086520255, "'": 0.0013725516619160771, 't': 0.0015102395555004478,
    #        '▁go': 0.0005502908607013524, '▁home': 0.0014923501294106245, '▁that': 0.0009823078289628029,
    #        '▁evening': 0.0018404676811769605, '▁and': 0.00046898151049390435, '▁': 0.09211429953575134,
    #        're': 0.0004362099862191826, 'sisted': 0.0027353481855243444, '▁Riley': 0.007774821948260069,
    #        's': 0.0036468186881393194, '▁attacks': 0.004170892760157585, '.': 0.0030060261487960815,
    #        '▁What': 0.002392962109297514, '▁does': 0.0019515401218086481, '▁need': 0.013643212616443634,
    #        '▁to': 0.0005810720031149685, '▁do': 0.0037925769574940205, '▁before': 0.00945215579122305,
    #        '▁this': 0.0017502920236438513, '?': 0.002131211804226041, '▁Options': 0.013728966936469078,
    #        'a': 0.004350507166236639, ')': 0.0063089970499277115, '▁Make': 0.05659395083785057,
    #        '▁new': 0.0028916974551975727, '▁plan': 0.0028365186881273985, ',': 0.013565066270530224,
    #        'b': 0.008598132990300655, '▁Go': 0.0036058370023965836, '▁see': 0.0018858902622014284}}
    # ]

    # attrs = [list(attr.values())[0] for attr in x]
    # # print(attrs)
    # tokens, attributions = list(attrs[0].keys()), list(attrs[0].values())
    # print(tokens)
    #
    # new_dict = {}
    #
    # for key, value in attrs[0].items():
    #     if key.startswith('_'):
    #         continue
    #     new_key = ''.join(filter(lambda x: x != '_', key))
    #     if new_key not in new_dict:
    #         new_dict[new_key] = value
    #     else:
    #         new_dict[new_key] += value
    #
    # print(new_dict)

    # filtered_tokens = [tokens[0]]
    # for word in tokens[1:]:
    #     print(word)
    #     if not word.startswith("▁"):
    #         print("here word:", word)
    #         filtered_tokens[-1] += f'{word}'
    #         # print("fil:", filtered_tokens)
    #     else:
    #         filtered_tokens.append(word)
    # print(filtered_tokens)
    #
    # attribution_score = [attributions[0]]
    # for idx, (word_idx, score) in enumerate(zip(word_map, attributions[1:])):
    #     if word_idx == word_map[idx + 1] and word_idx is not None:
    #         attribution_score[-1] = attribution_score[-1] + score
    #     else:
    #         attribution_score.append(score)
