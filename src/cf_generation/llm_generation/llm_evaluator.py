import os
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
from langchain import PromptTemplate
import jsonlines
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, T5Tokenizer, AutoTokenizer, AutoConfig

from src.few_shot.utils import save_to_disk

# from src.few_shot.generation import LLaMA

BASE_PATH="/storage/xyz/work/anon/research_projects/exp_calibration/"


def get_llama(model):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Set the seed
    torch.manual_seed(args.seed)

    # load squad data
    dataset = load_dataset("squad", "plain_text")
    train_data = dataset["train"]
    squad_data = [sample for sample in tqdm(train_data, total=len(train_data), desc="Loading SQuAD data ... ")]

    # prompt = "Answer the question from the context:"  # works but some modification testing
    # prompt = "Answer the question based on the context below. If the question cannot be answered " \
    #          "using the information provided, then answer with 'I don't know'."

    # prompt = "As an answer generator, your task is to generate a concise and clear answer " \
    #          "to the given question from the context."

    # prompt = "Answer the following question from the given context."  # does not work

    template = """ 
        Given the question: \n
        {query}
        Decide if the following retrieved context is relevant to the {answer}: \n
        {result}
        Answer in the following format: \n
        "Context is relevant: True or False." \n """.strip()

    GRADE_DOCS_PROMPT_FAST = PromptTemplate(input_variables=["query", "result", "answer"], template=template)

    device = torch.device('cuda:0')
    # model_name = "decapoda-research/llama-13b-hf"
    # model_name = "togethercomputer/GPT-NeoXT-Chat-Base-20B"
    model_name = "google/flan-ul2"
    # model = get_llama(model_name).to(device)
    # tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    # model.to(device)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = T5ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = LlamaForCausalLM.from_pretrained("huggyllama/llama-13b", torch_dtype=torch.bfloat16, device_map="auto")
    # tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-13b")
    # print(model.num_parameters())

    model_identifier = model_name.split("/")[-1]
    model_id = "alpaca"
    save_path = BASE_PATH + f"src/data/squad/{model_id}_qa_relevance_seed_{args.seed}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(BASE_PATH, f"src/data/squad/counterfactual_data_alpaca_13b_v2_qg_pipeline_all_data_cleaned.jsonl")
    # all_files = os.listdir(file_path)
    files = [file_path]
    # print(files)
    # generator = LLaMA(model, tokenizer)
    skipped = 0
    c = 0
    for file_name in files:
        examples = []
        with jsonlines.open(file_name) as reader:
            for example in tqdm(reader):
                try:
                    # c+=1
                    # if c <= 25000:
                    #     continue
                    id = example["id"].split("_")[0]
                    context = example["context"]
                    question = example["question"]
                    answer = example["answers"]["text"][0]

                    # print("Given ans:", example["answers"])

                    orig_example = [sample for sample in squad_data if sample["id"] == id][0]

                    orig_context = orig_example["context"]
                    orig_question = orig_example["question"]
                    orig_answer = orig_example["answers"]

                    # unmwated_words = ["question", "context", "answer"]
                    # # check if unmwated words are present in the question
                    # if any(word in question.lower() for word in unmwated_words):
                    #     continue

                    # process question
                    # sentences = raw_question.split("\nQ: ")[1].split("\n\n")
                    # # remove empty strings from the list
                    # sentences = [sentence for sentence in sentences if sentence]
                    # question = sentences[0]
                    # # remove ``` from the question
                    # question = question.replace("```", "")
                    # # remove new line characters
                    # question = question.replace("\n", "")

                    # input = f"\n\n{prompt} \n\nContext: {context} \n\nQuestion: {question} " \
                    #         f"\n\nAnswer: "  # does not work

                    # input = f"{prompt}\n C: {context}\n Q: {question}\n " \
                    #         f"A: "

                    # input = f"{prompt} \nContext: {orig_context} \nQuestion: {orig_question} " \
                    #         f"\nAnswer: {orig_answer['text'][0]}" \
                    #         f"\n{prompt} \nContext: {context} \nQuestion: {question} " \
                    #         f"\nAnswer: "

                    input = GRADE_DOCS_PROMPT_FAST.format(query=question, result=context, answer=answer)
                    # print(input)
                    input_ids = tokenizer.encode(input, return_tensors="pt").to(device)
                    # outputs = generator.generate([input], max_new_tokens=50, top_p=0.95)

                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids,
                            max_new_tokens=10,
                            temperature=0,
                            top_p=1,
                            top_k=40,
                            repetition_penalty=1.0,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # print(output)
                    # remove the context from the output
                    # output = output[len(input):]
                    # # print(output)
                    #
                    # answer = output.lstrip("\n").lstrip()
                    # answer = answer.replace("<bot>:", "")
                    # answer = answer.split("\n")[0]
                    # print("Answer: ", output)
                    c +=1
                    # if c==50:
                    #     break

                    result = {
                        "id": example["id"],
                        "question": question,
                        "context": context,
                        "answer": answer,
                        "context_relevance": output
                    }
                    # print(result)
                    # break
                    examples.append(result)
                    if c % 5000 == 0:
                        save_to_disk(
                            examples,
                            f"{save_path}counterfactual_samples_{model_id}_{c}.jsonl"
                        )
                        examples = []
                # break
                except Exception as e:
                    print("Skip")
        # save the remaining examples
        if examples:
            save_to_disk(
                examples,
                f"{save_path}counterfactual_samples_{model_id}_{c}.jsonl"
            )
