import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import jsonlines
from tqdm import tqdm
from datasets import load_dataset

from src.cf_generation.llm_generation.utils import save_to_disk

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"

# Set the seed
seed = 0
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Llama-2-13b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{model_name}")
model = AutoModelForCausalLM.from_pretrained(
    f"meta-llama/{model_name}"
)  # , torch_dtype=torch.bfloat16)
model.to(device)


def get_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(
        device
    )
    outputs = model.generate(**inputs, max_new_tokens=10,)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Set the seed
    torch.manual_seed(args.seed)

    # load squad data
    dataset = load_dataset("squad", "plain_text")
    train_data = dataset["train"]
    squad_data = [
        sample
        for sample in tqdm(
            train_data, total=len(train_data), desc="Loading SQuAD data ... "
        )
    ]
    test_model = "alpaca_13b"
    save_path = (
        BASE_PATH
        + f"src/data/squad/{model_name}_qa_relevance_{test_model}_seed_{args.seed}/"
    )
    # save_path = BASE_PATH + f"src/data/squad/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(
        BASE_PATH,
        # f"src/data/squad/gpt-4-0314_qa_relevance_seed_{args.seed}/counterfactual_samples_gpt-4-0314_llama_500.jsonl"
        f"src/data/squad/counterfactual_data_{test_model}_v2_qg_pipeline_all_data_cleaned.jsonl",
    )
    files = [file_path]
    skipped = 0
    c = 0
    for file_name in files:
        examples = []
        with jsonlines.open(file_name) as reader:
            for example in tqdm(reader):
                try:
                    c += 1

                    id = example["id"].split("_")[0]
                    context = example["context"]
                    question = example["question"]
                    answer = example["answers"]["text"][0]
                    orig_example = [
                        sample for sample in squad_data if sample["id"] == id
                    ][0]

                    orig_context = orig_example["context"]
                    orig_question = orig_example["question"]
                    orig_answer = orig_example["answers"]

                    # prompt = \
                    #     "You are a helpful assistant. Please think critically. "  \
                    #     f"Given the question: \n" \
                    #     f"{question} \n" \
                    #     f"Decide if the following retrieved context is relevant to answer this question: \n" \
                    #     f"{context} \n" \
                    #     "Answer in the following format: \n" \
                    #     "'Context is relevant: False or True.' \n" \
                    #     "Please answer False if you think that the question cannot be answered from the context. \n" \
                    #     "Assistant: Context is relevant: "

                    # prompt = \
                    #     f"Given the question: \n" \
                    #     f"{question} \n" \
                    #     f"Decide if the following retrieved context is relevant to answer the question: \n" \
                    #     f"{context} \n" \
                    #     "Your answer should be either True or False." \
                    #     "Assistant: "

                    # 2nd BEST TILL NOW (~75% accuracy)
                    ############################################################################################
                    # system_prompt = "You are a language/context evaluator. Your job is to identify whether " \
                    #                 "a given context is relevant for answering a given question."
                    #
                    # prompt = \
                    #     f"Given the question: \n" \
                    #     f"{question} \n" \
                    #     f"Decide if the following retrieved context is relevant to the {answer}: \n" \
                    #     f"{context} \n" \
                    #     "Your answer should be either True or False." \
                    #     "Assistant: "

                    #############################################################################################

                    # 2nd place
                    ############################################################################################
                    # system_prompt = "You are a language/context evaluator. Your job is to identify whether " \
                    #                 "a given context is relevant for answering a given question."
                    #
                    # prompt = \
                    #     f"Given the question: \n" \
                    #     f"{question} \n" \
                    #     f"Decide if the following retrieved context is relevant to the question: \n" \
                    #     f"{context} \n" \
                    #     "Your answer should be either True or False." \
                    #     "Assistant: "

                    #############################################################################################

                    # BEST TILL NOW(~81 % accuracy)
                    ############################################################################################
                    system_prompt = (
                        "You are a language/context evaluator. Your job is to identify whether "
                        "a given context is relevant for answering a given question."
                    )

                    task_prompt = (
                        f"Given the question: \n"
                        f"{question} \n"
                        f"Decide if the following retrieved context is relevant to the {answer}: \n"
                        f"{context} \n"
                        "Answer in the following format:\n True or False."
                        "Assistant: "
                    )

                    #############################################################################################

                    template = (
                        f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
                        f"{task_prompt}"
                        "[/INST]"
                    )

                    # print(template)
                    # break
                    # print("GPT-4 answer: ", example["context_relevance"]["choices"][0]["message"]["content"])
                    output = get_response(prompt=template)
                    output = output[len(template) :].strip().split("\n")[0]
                    # print("LLAMA answer: ", output)
                    # print("----"*10)

                    result = {
                        "id": example["id"],
                        "question": question,
                        "context": context,
                        "answer": answer,
                        "context_relevance": output,
                    }
                    # print(result)
                    # break
                    examples.append(result)
                    if c % 5000 == 0:
                        save_to_disk(
                            examples,
                            f"{save_path}counterfactual_samples_{model_name}_{test_model}_{c}.jsonl",
                        )
                        examples = []
                        print(f"----Saved {c} samples----")
                    # if c == 200:
                    #     break
                except Exception as e:
                    print("Skip")
        # save the remaining examples
        if examples:
            save_to_disk(
                examples,
                f"{save_path}counterfactual_samples_{model_name}_{test_model}_{c}.jsonl",
            )
