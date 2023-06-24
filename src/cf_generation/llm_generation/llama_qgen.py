import os
import torch
from transformers import LLaMAForCausalLM
from transformers.models.llama.tokenization_llama import LLaMATokenizer

import jsonlines
from tqdm import tqdm
from datasets import load_dataset

from src.few_shot.utils import save_to_disk

BASE_PATH="/storage/ukp/work/sachdeva/research_projects/exp_calibration/"


def get_llama(model):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = LLaMAForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


if __name__ == '__main__':
    # load squad data
    dataset = load_dataset("squad", "plain_text")
    train_data = dataset["train"]
    squad_data = [sample for sample in tqdm(train_data, total=len(train_data), desc="Loading SQuAD data ... ")]

    c = 0
    examples = []

    prompt = "Generate a fluent and answerable question from the given context. Ensure that the answer " \
             "is a span in the context and is less than 10 words."
    # prompt = "Please generate a question that can be answered from the given context. Here is an example:" \

    device = torch.device('cuda:0')
    model_name = "llama-13b-hf"
    model = get_llama(model_name).to(device)
    tokenizer = LLaMATokenizer.from_pretrained(model_name)
    # print(model.num_parameters())

    model_identifier = model_name.split("/")[-1]
    save_path = BASE_PATH + f"src/data/squad/{model_identifier}_qg_1/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    """
    How are you doing today?

    I asked **LlaMA**: "How are you doing today?"
    And got this hilarious reply

    How are you doing today? Good to see you!
    I know, we haven't seen each other in awhile, but that's okay. Things are different now.
    I am older and wiser, and you are older and wiser. We have both grown so much in the past few years,
    and we have so much to share with each other. I really want us to be friends.
    I know, you're thinking that I'm just another girl who wants to get close to you for what you can
    give me. I do want to be close to you, but I don't want anything from you. I have my own life, my own
    career, my own friends, and my own family. I have everything I want and need. I'm not looking for you
    to make me happy, because I know that I can do that all by myself. I just want to have a good
    relationship with you, and I think that we can

    """
    skipped_instances = 0
    model.eval()
    with jsonlines.open(BASE_PATH + "src/data/squad/squad_counterfactuals_noise_min_filtered_final_2.jsonl") as reader:
        for example in tqdm(reader):
            try:
                c += 1
                if c<=60000:
                    continue
                id = example["id"].split("_")[0]
                context = example["context"]
                orig_example = [sample for sample in squad_data if sample["id"] == id][0]
                # print(orig_example)
                orig_context = orig_example["context"]
                orig_question = orig_example["question"]
                orig_answer = orig_example["answers"]

                input = f"{prompt} \n\nC: {context} \n\nQ: "

                input_ids = tokenizer.encode(input, return_tensors="pt").to(device)
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids,
                        # do_sample=True,
                        # max_length=1024,
                        max_new_tokens=50,
                        # top_p=0.95,
                        # temperature=0.8,
                    )
                outputs = tokenizer.decode([el.item() for el in generated_ids[0]])
                sentences = outputs.split("\nQ: ")[1].split("\n\n")
                # remove empty strings from the list
                sentences = [sentence for sentence in sentences if sentence]
                question = sentences[0]
                # remove ``` from the question
                question = question.replace("```", "")
                # remove new line characters
                question = question.replace("\n", "")
                # print("Actual question: ", question)
                # print("-" * 100)

                result = {
                    "id": example["id"],
                    "question": question,
                    "context": context
                }

                examples.append(result)
                if c % 5000 == 0:
                    save_to_disk(
                        examples,
                        f"{save_path}counterfactual_questions_{model_identifier}_{c}.jsonl"
                    )
                    examples = []

            except Exception as e:
                print(outputs)
                skipped_instances += 1
                continue

        # save the remaining examples
        if examples:
            save_to_disk(
                examples,
                f"{save_path}counterfactual_questions_{model_identifier}_{c}.jsonl"
            )
        print(f"Skipped instances: {skipped_instances}")

