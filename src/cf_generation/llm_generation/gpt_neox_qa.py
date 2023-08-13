import os

from transformers import AutoModelForCausalLM, T5Tokenizer, AutoTokenizer, AutoConfig
import torch
import jsonlines
from tqdm import tqdm
from datasets import load_dataset

# from src.few_shot.utils import save_to_disk

BASE_PATH="/storage/ukp/work/sachdeva/research_projects/exp_calibration/"
# BASE_PATH = "/home/sachdeva/projects/ukp/exp_calibration/"

def save_to_disk(data, file_name):
    with jsonlines.open(file_name, "a") as writer:
        for example in tqdm(data, total=len(data), desc="Saving samples ... "):
            writer.write(example)


if __name__ == '__main__':

    # load squad data
    dataset = load_dataset("squad", "plain_text")
    train_data = dataset["train"]
    squad_data = [sample for sample in tqdm(train_data, total=len(train_data), desc="Loading SQuAD data ... ")]

    c = 0
    examples = []

    # prompt = "Generate a fluent and answerable question from the given context. Ensure that the answer " \
    #          "is a span in the context and is less than 10 words."
    # prompt = "Given the context below, generate a fluent question that can be answered from it."  # does not work
    # prompt = "Using the provided context, generate a coherent question that can " \
    #          "be answered based on the information within the context."   # okay to use

    # prompt = "Given the context below, generate a question that can be answered using information " \
    #          "directly stated in the context."   # does not work

    # prompt = "Using the given context, generate a question that requires selecting a short " \
    #          "and specific answer from it."

    # prompt = "You are a question generation model. Given the context below, please generate a question that " \
    #          "can be answered based on the information given in the context."  # use this

    # prompt = "As a question generation model, your task is to generate a relevant and informative question " \
    #          "based on the given context. The question should be of type what, where, which, why, who, how, etc., " \
    #          "and should be answerable based on the information provided in the context. Please ensure that the " \
    #          "generated question is clear and concise, providing specific details that allow for an accurate " \
    #          "response. Additionally, please take into account any relevant background or contextual information " \
    #          "when generating your question."

    # prompt = "Your task is to create a question that can only be answered by identifying a specific " \
    #          "span of text within a given context. The question you generate should require the reader " \
    #          "to identify key details or information contained within the specified span of text. This " \
    #          "could include facts, figures, names, dates, events, or other relevant information. Please make " \
    #          "sure that the question is challenging but not impossible to answer based on the given context. " \
    #          "It should also be clear and concise in order to avoid confusion."

    # prompt = "As a question generation model, your task is to generate a relevant and answerable question " \
    #          "based on the given context. Please carefully read the context provided and craft a question that " \
    #          "can be answered using information from the text. Your question should be clear and concise, with " \
    #          "specific focus on key details or concepts within the context. You should also ensure that your " \
    #          "question is open-ended enough to allow for various possible answers while still being grounded in " \
    #          "the information presented in the context. Please note that you should aim to create questions that " \
    #          "encourage critical thinking and deeper understanding of the topic at hand."

    # prompt = "As a question generation model, your task is to generate a relevant and answerable question " \
    #          "based on the given context. Please carefully read the context provided and craft a question that " \
    #          "can be answered using information from the text. Your question should be clear and concise, with " \
    #          "specific focus on key details or concepts within the context. You should also ensure that your " \
    #          "question should allow one possible answer while still being grounded in " \
    #          "the information presented in the context. Please note that you should aim to create questions that " \
    #          "encourage critical thinking and deeper understanding of the topic at hand."  # nope

    # prompt = "Your task is to create a concise and clear question that can be answered from the given context. " \
    #          "The question should be fluent and answerable, with an answer span of less than 10 words. Please " \
    #          "ensure that the question is relevant to the context provided and encourages critical thinking or " \
    #          "analysis. Your response should demonstrate your ability to distill key information into a focused " \
    #          "and effective question."  # nope

    # prompt = "Your task is to generate a question that can be answered by an answer span within a " \
    #          "given context. The context should be a piece of text, such as a news article or historical " \
    #          "document, and the question should require understanding and analysis of the information presented " \
    #          "in the context. The generated question should be clear and concise, focusing on key details or events " \
    #          "described in the context. It should also be specific enough to have a single correct answer that " \
    #          "can be found within the context."   # good prompt but unanswerable questions

    # prompt = "As a question generation model, your task is to generate a question from a given context that" \
    #          " can be answered by a specific span of text highlighted in the context. The question should be " \
    #          "clear and concise, using appropriate language and grammar. It should also focus on extracting " \
    #          "relevant information from the highlighted span. Please note that you should be able to handle " \
    #          "various types of contexts, including those with complex sentence structures or technical jargon. " \
    #          "Your response should provide a well-formed question that accurately captures the meaning of the " \
    #          "highlighted span and encourages critical thinking."  # okayish

    # prompt = "Using the given context, create a question that is grammatically correct and coherent. The answer " \
    #          "to the question should be a specific span of text found within the context. Please ensure that your " \
    #          "question is clear and concise, and that it accurately reflects the information provided in the context." # okayish


    # prompt = "Create a question based on the context provided below that elicits a fluent and " \
    #          "concise answer from within the given text. Please provide a clear and specific question " \
    #          "that can be answered directly by a single span of text within the passage, without " \
    #          "requiring additional information or interpretation. The question should also be relevant " \
    #          "to the content of the passage and demonstrate an understanding of its key points."
    # Note " \
    #          "that your question should encourage creativity in generating unique and interesting " \
    #          "questions while still maintaining focus on accuracy and relevance."   # nope

    # Context:\n some text \nQuestion: What is the question?
    # prompt = "Generate a fluent and answerable question from the given context. Ensure that the answer " \
    #          "is a span in the context and is less than 10 words."

    # prompt = "You are a question generation model. Given the context below, please generate an answerable question " \
    #     "from it."

    # prompt = "Your task is to generate a fluent and coherent question based on the given context, " \
    #          "which should strictly be answerable by a span within the context. The question should " \
    #          "take inspiration from extractive question answering datasets like SQuAD. Please ensure " \
    #          "that your question is clear and concise, with a focus on accuracy and relevance. Your " \
    #          "question should also demonstrate an understanding of the key details and concepts presented " \
    #          "in the context, while encouraging critical thinking and analysis. Please note that your " \
    #          "response should allow for various relevant and creative questions."  # nope

    prompt = "Given the context please answer the question."

    model_name = "EleutherAI/gpt-neox-20b"
    model_identifier = model_name.split("/")[-1]
    save_path = BASE_PATH + f"src/data/squad/{model_identifier}_qg"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config = AutoConfig.from_pretrained(model_name, pad_token_id=0, eos_token_id=0)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.eval()
    skipped_instances = 0
    with jsonlines.open(
            BASE_PATH +
            "src/data/squad/counterfactual_samples_Llama-2-13b-chat-hf_gpt_neox_context_filtered_complete.jsonl") as reader:
        for example in tqdm(reader):
            try:
                c += 1
                if c==20:
                    break
                # print(c)
                id = example["id"].split("_")[0]
                context = example["context"]
                question = example["question"]
                orig_example = [sample for sample in squad_data if sample["id"] == id][0]
                # print(orig_example)
                orig_context = orig_example["context"]
                orig_question = orig_example["question"]
                orig_answer = orig_example["answers"]

                input = f"{prompt} \nContext: {context} \nQuestion: {question} \nAnswer: "   # use this one

                # input = f"{prompt}\n Context: {context}\n Question: "

                # print(input)
                inputs = tokenizer(input, return_tensors="pt").to("cuda")
                generated_ids = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=20,
                    # temperature=0.5,
                )
                outputs = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                answer = outputs[len(input):].strip().split("\n")[0]
                print("pred: ", answer)
                print("orig: ", example["answers"])
                print("-"*100)

                # sentences = outputs.split("\nQuestion: ")[1].split("\n\n")
                # # remove empty strings from the list
                # sentences = [sentence for sentence in sentences if sentence]
                # question = sentences[0]
                # # remove ``` from the question
                # question = question.replace("```", "")
                # # remove new line characters
                # question = question.replace("\n", "")
                # print("Context: ", context)
                # print("Original outputs: ", outputs)
                # print("Actual question: ", question)
                # print("-"*100)


                # if c == 20:
                #     break
                # break
                # result = {
                #     "id": example["id"],
                #     "question": question,
                #     "context": context,
                #     "answer": answer
                # }
                # # print(result)
                # examples.append(result)
                # if c % 5000 == 0:
                #     save_to_disk(
                #         examples,
                #         f"{save_path}/counterfactual_questions_{model_identifier}_{c}.jsonl"
                #     )
                #     examples = []

            except Exception as e:
                # print(outputs)
                skipped_instances += 1
                continue

        # save the remaining examples
        # if examples:
        #     save_to_disk(
        #         examples,
        #         f"{save_path}/counterfactual_questions_{model_identifier}_{c}.jsonl"
        #     )
