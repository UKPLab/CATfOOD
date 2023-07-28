import os
import glob
import json
import jsonlines
from tqdm import tqdm
from datasets import load_dataset

import spacy

# Load the Spacy model
nlp = spacy.load('en_core_web_sm')
stop_words = set(spacy.lang.en.stop_words.STOP_WORDS)


BASE_PATH="/storage/ukp/work/sachdeva/research_projects/exp_calibration/"
# BASE_PATH = "/home/sachdeva/projects/ukp/exp_calibration/"

def collate_jsonl_files(data_path=None, save_path=None):

    if not data_path:
        # Set the path to the directory containing the JSONL files
        data_path = BASE_PATH + "src/data/squad/t5_squad_counterfactuals/final_cfs"
    print("Loading data from: ", data_path)

    if not save_path:
        # Set the path to the output file
        save_path = BASE_PATH + "src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_min_filtered_diff_ans_final.jsonl"
    print("Saving data to: ", save_path)

    # Create an empty list to store the samples
    samples = []

    # Loop over all the JSONL files in the directory
    for file_path in glob.glob(data_path + "/*.jsonl"):
        # Open the JSONL file and read the samples
        with jsonlines.open(file_path) as reader:
            samples.extend(reader)

    # Write the collated samples to the output file
    with jsonlines.open(save_path, mode="w") as writer:
        writer.write_all(samples)


def save_to_disk(data, file_name):
    with jsonlines.open(file_name, "a") as writer:
        for example in tqdm(data, total=len(data), desc="Saving samples ... "):
            writer.write(example)


def get_answer_start(question, context, answer):
    # remove trailing punctuation from answer
    answer = answer.rstrip(".")
    if answer == "":
        return -1
    # Find all occurrences of the string in the context
    indices = []
    end_index = 0
    answer_start = None
    context = context.lower()
    answer = answer.lower()

    while True:
        index = context.find(answer, end_index)
        if index == -1:
            break
        indices.append(index)
        end_index = index + len(answer)
    # print("Indices: ", indices)
    if not indices:
        return -1
    if len(indices)>1:
        answer_indices = [i for i in range(len(context)) if context.startswith(answer, i)]
        # print(answer_indices)
        # Choose the correct instance of the answer based on the question and context
        doc = nlp(context)
        question_tokens = set([token.text.lower() for token in nlp(question) if not token.is_stop])
        max_match_count = 0
        max_match_sent = None
        # print(doc)
        for sent in doc.sents:    # print(context.find(a    # print(context.find(answer))nswer))
            sent_tokens = set([token.text.lower() for token in sent if not token.is_stop])
            match_count = len(question_tokens.intersection(sent_tokens))
            # print(sent, match_count)
            if match_count > max_match_count:
                max_match_count = match_count
                max_match_sent = sent
        if max_match_sent is not None:
            if answer.lower() in max_match_sent.text.lower():
                for index in answer_indices:
                    # get the start and end of sentence as in context
                    if max_match_sent.start_char <= index < max_match_sent.end_char:
                        answer_start = index
                        break
        if answer_start is None:
            answer_start = answer_indices[0]
    else:
        answer_start = indices[0]

    return answer_start


def _preprocess_llama_questions():
    current_files = []
    model_name = "decapoda-research/llama-13b-hf"
    model_identifier = model_name.split("/")[-1]
    all_files = os.listdir(BASE_PATH + f"src/data/squad/{model_identifier}_qg/")
    files = [file for file in all_files if file not in current_files]
    print(files)

    c = 0
    for file in files:
        examples = []
        with jsonlines.open(BASE_PATH + f"src/data/squad/{model_identifier}_qg/" + file) as reader:
            for example in tqdm(reader):
                id = example["id"].split("_")[0]
                context = example["context"]
                raw_question = example["question"]
                print("*" * 100)
                print(raw_question)
                print("-" * 100)

                # process question
                sentences = raw_question.split("\nQ: ")[1].split("\n\n")
                # remove empty strings from the list
                sentences = [sentence for sentence in sentences if sentence]
                question = sentences[0]
                # remove ``` from the question
                question = question.replace("```", "")
                # remove new line characters
                question = question.replace("\n", "")

                print(question)
                print("*"*100)
                c += 1
                if c == 10:
                    break
        break

def remove_duplicate_examples(input_file, output_file):
    seen_examples = set()

    with jsonlines.open(output_file, "w") as writer:
        with jsonlines.open(input_file) as reader:
            for example in tqdm(reader):
                qa_string = example["question"] + example["answers"]["text"][0]
                if qa_string not in seen_examples:
                    writer.write(example)
                    seen_examples.add(qa_string)


def remaining_samples(complete_file, subset_file, save_path):
    """
    Read the complete file and subset file and list ids not present
    in the subset file
    """
    with jsonlines.open(complete_file) as reader:
        all_ids = [sample["id"] for sample in reader]
    with jsonlines.open(subset_file) as reader:
        subset_ids = [sample["id"] for sample in reader]
    remaining_ids = [idx for idx in all_ids if idx not in subset_ids]
    with jsonlines.open(save_path, "w") as writer:
        with jsonlines.open(complete_file) as reader:
            for example in tqdm(reader):
                idx = example["id"]
                if idx in remaining_ids:
                    writer.write(example)
    # return remaining_ids


def filter_counterfactuals():
    """
    compare generated answer to the answers generated via 3 random seeds
    """
    data_path = os.path.join(BASE_PATH, "src/data/squad")
    orig_ans_path = os.path.join(data_path, "flan-t5-xxl-v3_collated_data_with_answers_processed.jsonl")
    answers = {}
    seeds = [0, 1, 42]
    for seed in seeds:
        path = os.path.join(data_path, f"few_shot_flan-t5-xxl_qa_eval_seed_{seed}/counterfactual_samples_flan-t5-xxl_1.jsonl")
        with jsonlines.open(path) as reader:
            for sample in tqdm(reader):
                idx = sample["id"]
                answer = sample["answer"]
                if idx in answers.keys():
                    answers[idx].append(answer.lower())
                else:
                    answers[idx] = [answer.lower()]
                # break
        # break

    with jsonlines.open(orig_ans_path) as reader:
        examples = []
        for sample in tqdm(reader):
            target_answer = sample["answers"]["text"][0]
            idx = sample["id"]
            if idx in answers.keys():
                # compare answers
                ans_count = answers[idx].count(target_answer.lower())
                if ans_count >= 2:
                    examples.append(sample)

    save_to_disk(examples, os.path.join(data_path, "flan_t5_xxl_collated_data_with_answers_processed_filtered.jsonl"))
    # print(ans_count)


def context_eval():
    data_path = os.path.join(BASE_PATH, "src/data/squad")
    # counterfactual_data_gpt_neox_20b_v2_qg_pipeline_all_data_cleaned.jsonl
    # counterfactual_data_gpt_jt_v2_qg_pipeline_all_data_cleaned.jsonl
    # counterfactual_data_llama_13b_v1_qg_pipeline_all_data_cleaned.jsonl
    # counterfactual_data_alpaca_13b_v2_qg_pipeline_all_data_cleaned.jsonl
    # flan-t5-xxl-v3_collated_data_with_answers_processed.jsonl
    # flan_ul2_collated_data_with_answers_processed.jsonl
    orig_ans_path = os.path.join(data_path, "counterfactual_data_gpt_neox_20b_v2_qg_pipeline_all_data_cleaned.jsonl")
    answers = {}

    # context rel file
    path = os.path.join(data_path, f"counterfactual_samples_Llama-2-13b-chat-hf_gpt_neox_complete.jsonl")
    with jsonlines.open(path) as reader:
        for sample in tqdm(reader):
            # print(sample)
            idx = sample["id"]
            context_rel = sample["context_relevance"]
            if idx in answers.keys():
                answers[idx].append(context_rel)
            else:
                answers[idx] = [context_rel]
            # break
    # break
    # print(list(answers.values())[:10])
    c = 0
    with jsonlines.open(orig_ans_path) as reader:
        examples = []
        for sample in tqdm(reader):
            # target_answer = sample["answers"]["text"][0]
            idx = sample["id"]
            if idx in answers.keys():
                context_rel = answers[idx][0]
                if context_rel.__contains__("True"):
                    examples.append(sample)
                    c+=1
    print(c)
    save_to_disk(
        examples,
        os.path.join(data_path, "counterfactual_samples_Llama-2-13b-chat-hf_gpt_neox_context_filtered_complete.jsonl")
    )


def openai_context_eval():
    data_path = os.path.join(BASE_PATH, "src/data/squad")
    eval_model = "gpt-4-0314"
    test_model = "flan_ul2"
    save_path = BASE_PATH + f"src/data/squad/{eval_model}_qa_relevance_seed_0/"
    orig_ans_path = os.path.join(save_path, f"counterfactual_samples_{eval_model}_{test_model}_500.jsonl")
    answers = {}

    # path = os.path.join(data_path, f"counterfactual_samples_gpt_jt_context_relevance.jsonl")
    # with jsonlines.open(path) as reader:
    #     for sample in tqdm(reader):
    #         idx = sample["id"]
    #         context_rel = sample["context_relevance"]
    #         if idx in answers.keys():
    #             answers[idx].append(context_rel)
    #         else:
    #             answers[idx] = [context_rel]
            # break
    # break
    # print(list(answers.values())[:10])
    c = 0
    with jsonlines.open(orig_ans_path) as reader:
        examples = []
        for sample in tqdm(reader):
            # target_answer = sample["answers"]["text"][0]
            idx = sample["id"]
            output = sample["context_relevance"]
            context_rel = output["choices"][0]["message"]["content"]
            if context_rel.__contains__("True"):
                c+=1
            # print(context_rel)
            # break
    print(c)


    #         if idx in answers.keys():
    #             context_rel = answers[idx][0]
    #             if context_rel in ["True", "yes"]:
    #                 examples.append(sample)
    #                 c+=1
    # print(c)
    # save_to_disk(examples, os.path.join(data_path, "gpt_jt_collated_data_with_answers_processed_context_relevance.jsonl"))


def context_noise_filter():
    data_path = os.path.join(BASE_PATH, "src/data/squad")
    rel_answers = {}
    seed_answers = {}

    orig_ans_path = os.path.join(data_path, "flan-t5-xxl-v3_collated_data_with_answers_processed.jsonl")
    seeds = [0, 1, 42]
    for seed in seeds:
        path = os.path.join(data_path, f"few_shot_flan-t5-xxl_qa_eval_seed_{seed}/counterfactual_samples_flan-t5-xxl_1.jsonl")
        with jsonlines.open(path) as reader:
            for sample in tqdm(reader):
                idx = sample["id"]
                answer = sample["answer"]
                if idx in seed_answers.keys():
                    seed_answers[idx].append(answer.lower())
                else:
                    seed_answers[idx] = [answer.lower()]

    path = os.path.join(data_path, f"counterfactual_samples_flan_t5_xxl_context_relevance.jsonl")
    with jsonlines.open(path) as reader:
        for sample in tqdm(reader):
            idx = sample["id"]
            context_rel = sample["context_relevance"]
            if idx in rel_answers.keys():
                rel_answers[idx].append(context_rel)
            else:
                rel_answers[idx] = [context_rel]
            # break
    # break
    # print(list(answers.values())[:10])
    c = 0
    with jsonlines.open(orig_ans_path) as reader:
        examples = []
        for sample in tqdm(reader):
            target_answer = sample["answers"]["text"][0]
            idx = sample["id"]
            if idx in rel_answers.keys():
                context_rel = rel_answers[idx][0]
                if context_rel in ["True", "yes"]:
                    examples.append(sample)
                    c += 1
                elif idx in seed_answers.keys():
                    # compare answers
                    ans_count = seed_answers[idx].count(target_answer.lower())
                    if ans_count >= 2:
                        examples.append(sample)
                        c += 1
    print(c)
    save_to_disk(examples,
                 os.path.join(data_path, "flan_t5_xxl_collated_data_with_answers_processed_context_relevance_noise_filter.jsonl"))


def compare_closed_open():
    closed_model = "gpt-4-0314"
    open_model = "Llama-2-13b-chat-hf"
    test_model = "llama"
    data_path = os.path.join(BASE_PATH, "src/data/squad")
    open_path = os.path.join(data_path, f"{open_model}_qa_relevance_seed_0/counterfactual_samples_{open_model}_{test_model}_500_2.jsonl")
    closed_path = os.path.join(data_path, f"{closed_model}_qa_relevance_seed_0/counterfactual_samples_{closed_model}_{test_model}_500.jsonl")

    c = 0

    with jsonlines.open(closed_path) as reader:
        closed_examples = [sample for sample in tqdm(reader)]

    with jsonlines.open(open_path) as reader:
        open_examples = [sample for sample in tqdm(reader)]

    for closed, open in zip(closed_examples, open_examples):
        closed_ans = closed["context_relevance"]["choices"][0]["message"]["content"]
        open_ans = open["context_relevance"]

        if closed_ans.__contains__("True") and open_ans.__contains__("True"):
            c+=1
        if closed_ans.__contains__("False") and open_ans.__contains__("False"):
            c+=1

    print(c)


def save_irrelevant_samples():
    data_path = os.path.join(BASE_PATH, "src/data/squad")
    path = os.path.join(data_path, f"flan_ul2_collated_data_with_answers_processed_context_irrelevance.jsonl")
    with jsonlines.open(path) as reader:
        samples = [sample for sample in tqdm(reader)]
    import pandas as pd
    from random import shuffle
    import random
    print(samples[:10])

    shuffled_dataset = sorted(samples, key=lambda k: random.random())
    print(type(shuffled_dataset))
    df = pd.DataFrame()
    print(shuffled_dataset[:10])
    df["question"] = [sample["question"] for sample in shuffled_dataset[:100]]
    df["context"] = [sample["context"] for sample in shuffled_dataset[:100]]
    df["answer"] = [sample["answers"]["text"][0] for sample in shuffled_dataset[:100]]
    print(df.head())

    df.to_excel(os.path.join(data_path, f"flan_ul2_irrelevant.xlsx"))


def select_noise_relevant_samples():
    data_path = os.path.join(BASE_PATH, "src/data/squad")
    relevant_path = os.path.join(data_path, "flan_ul2_collated_data_with_answers_processed_context_relevance.jsonl")
    examples = []
    noise_path = os.path.join(data_path, f"flan_ul2_collated_data_with_answers_processed_filtered.jsonl")
    with jsonlines.open(noise_path) as reader1:
        noise_idx = [ex["id"] for ex in tqdm(reader1)]
    with jsonlines.open(relevant_path) as reader2:
            for example in tqdm(reader2):
                if example["id"] in noise_idx:
                    examples.append(example)

    save_to_disk(examples,
                 os.path.join(data_path, "flan_ul2_collated_data_with_answers_processed_context_relevance_filtered.jsonl"))


def agg_answers():
    paths = ["t5-large-squad-qa-seed-42", "t5-large-squad-qa-seed-0", "t5-large-squad-qa-seed-1"]
    c = 0
    data_path = []
    for path in paths:
        c+=1
        data_path.append(BASE_PATH + f"src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_nf_{path}_1.jsonl")
    with jsonlines.open(BASE_PATH + f"src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_filtered.jsonl") as reader:
        examples1 = [ex for ex in reader]
    # with jsonlines.open(data_path[0]) as reader:
    #     examples1 = [ex for ex in reader]
    # with jsonlines.open(data_path[1]) as reader:
    #     examples2 = [ex for ex in reader]
    # with jsonlines.open(data_path[2]) as reader:
    #     examples3 = [ex for ex in reader]
    # min filter
    with jsonlines.open(BASE_PATH + f"src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_mf.jsonl") as reader:
        min_ex = [ex for ex in reader]

    min_ex_ids = [ex["id"] for ex in min_ex]
    for i, ex1 in tqdm(enumerate(examples1)): # examples2, examples3, min_ex)):
        # ex1["alternate_answers"].append(ex2["alternate_answers"][0])
        # ex1["alternate_answers"].append(ex3["alternate_answers"][0])
        if ex1["id"] in min_ex_ids:
            idx = min_ex_ids.index(ex1["id"])
            ex1["similarity"] = min_ex[idx]["similarity"]
    save_to_disk(
        examples1,
        BASE_PATH + f"src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_filtered_final.jsonl"
    )


def _add_answer_to_noise_filtered():
    with jsonlines.open(
            BASE_PATH + f"src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_min_filtered_dedup.jsonl") as reader:
        nf_samples = [ex for ex in reader]
        nf_idx = [ex["id"] for ex in nf_samples]

    with jsonlines.open(
            BASE_PATH + f"src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_filtered_final.jsonl") as reader:
        all_samples = [ex for ex in reader]

    examples = []
    for ex in tqdm(all_samples):
        if ex["id"] not in nf_idx:
            continue
        # if ex["id"] in nf_idx:
        examples.append(ex)
    print(len(examples))
    save_to_disk(
        examples,
        BASE_PATH + f"src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_min_filtered_dedup_final.jsonl"
    )


if __name__ == "__main__":

    ############ FILTERING ############
    # complete_file = f"{BASE_PATH}src/data/squad/squad_counterfactuals_28_03.jsonl"
    # subset_file = f"{BASE_PATH}src/data/squad/GPT-NeoXT-Chat-Base-20B-v1_collated_data_with_answers_processed.jsonl"
    # save_path = f"{BASE_PATH}src/data/squad/GPT-NeoXT-Chat-Base-20B-v1_remaining_counterfactuals.jsonl"
    # remaining_samples(complete_file, subset_file, save_path)

    # remove_duplicate_examples(subset_file, save_path)
    # filter_counterfactuals()
    context_eval()
    # select_noise_relevant_samples()
    # collate_jsonl_files()
    # _add_answer_to_noise_filtered()

    # context_noise_filter()

    ############ REMOVE DUPLICATES ############
    # remove_duplicate_examples(
    #     input_file=f"{BASE_PATH}src/data/squad/counterfactual_data_llama_13b_v1_qg_pipeline_all_data.jsonl",
    #     output_file=f"{BASE_PATH}src/data/squad/counterfactual_data_llama_13b_v1_qg_pipeline_all_data_cleaned.jsonl"
    # )

    # dataset = load_dataset("squad", "plain_text")
    # train_data = dataset["train"]
    # squad_data = [sample for sample in tqdm(train_data, total=len(train_data), desc="Loading SQuAD data ... ")]
    # c = 0
    # # read the jsonl file
    # files = [
    #     "t5_squad_counterfactuals/rag_counterfactuals_with_answers_squad_qg_1.jsonl"
    #     # "t5_squad_counterfactuals/rag_counterfactuals_complete_noise_min_filtered_final_dedup_1.jsonl",
    #     #  "llama_collated_data_with_answers_processed_context_relevance.jsonl",
    #      # "gpt_neox_collated_data_with_answers_processed_context_relevance.jsonl",
    #      # "flan_ul2_collated_data_with_answers_processed_context_relevance_noise_filter.jsonl"
    # ]
    # file_paths = [f"{BASE_PATH}src/data/squad/{name}" for name in files]
    #
    # # for path in file_paths:
    # #     with jsonlines.open(path) as reader:
    # #         ids = [example["id"].split("_")[0] for example in tqdm(reader)]
    #
    # common_ids = None
    #
    # for path in file_paths:
    #     with jsonlines.open(path) as reader:
    #         ids = set([example["id"].split("_")[0] for example in tqdm(reader)])
    #
    #         if common_ids is None:
    #             common_ids = ids
    #         else:
    #             common_ids = common_ids.intersection(ids)
    #
    # common_ids = list(common_ids)
    # print(len(common_ids))
    #
    # for q_id in common_ids:
    #     if q_id == "5726a49df1498d1400e8e5d5":
    #         c += 1
    #         # if c<200:
    #         #     continue
    #         orig_example = [sample for sample in squad_data if sample["id"] == q_id][0]
    #         orig_question = orig_example["question"]
    #         # if orig_question == "What did John Rawls publish?":
    #         print("id: ", q_id)
    #         print("Original question: ", orig_question)
    #         for i, path in enumerate(file_paths):
    #             with jsonlines.open(path) as reader:
    #                 for example in tqdm(reader):
    #                     id = example["id"].split("_")[0]
    #                     context = example["context"]
    #                     answer = example["answers"]
    #                     if q_id == id:
    #                         # question = example["question"]
    #                         # print("Model: ", files[i].split("_")[0])
    #                         # print("Cf: ", question)
    #                         # print("context: ", context)
    #                         # print("answer: ", answer)
    #                         print(example)
    #         print("-"*100)
    #         break
    #     else:
    #         continue

    # compare_closed_open()

    # model = "gpt_neox"
    # collate_jsonl_files(
    #     data_path=os.path.join(BASE_PATH, f"src/data/squad/Llama-2-13b-chat-hf_qa_relevance_{model}_seed_0/"),
    #     save_path=os.path.join(BASE_PATH, f"src/data/squad/counterfactual_samples_Llama-2-13b-chat-hf_{model}_complete.jsonl")
    # )
