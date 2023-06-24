import torch
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import traceback
from src.calibration.baseline import dataloader, utils
from sklearn.decomposition import PCA



BASE_PATH = "/storage/xyz/work/anon/research_projects/exp_calibration/"


class DenseRepresentations:
    def __init__(self, model, tokenizer):
        self.model= model
        self.tokenizer = tokenizer

    def extract_representations(self, inputs, pca=True):

        input_ids = self.tokenizer.encode(
            inputs,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            )
        # Get the start and end logits
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            hidden_states = outputs.hidden_states[0]
        # print("hidden:", hidden_states.shape)

        # # Get the answer span with the highest probability
        # start_idx, end_idx = torch.argmax(start_logits), torch.argmax(end_logits)
        # answer_span = inputs['input_ids'][0][start_idx:end_idx + 1]
        # answer_states = hidden_states[:, start_idx:end_idx + 1, :]
        # print(answer_states)
        # print(answer_states.shape)
        #
        # # Calculate the dense representation of the answer span
        # answer_rep = torch.mean(answer_states, dim=1)
        # print(answer_rep)

        # get model attributions
        # attributions = utils.load_bin(f"{BASE_PATH}src/data/squad_adversarial/attn_info_flan_ul2_context_noise_rel.bin")
        # print(attributions[ex["id"]])
        # print(attributions[ex["id"]]["attributions"].shape)
        # # get the indices that would sort the array
        # sorted_indices = np.argsort(attributions[ex["id"]]["attributions"])
        #
        # # get the top k indices
        # k = 10
        # top_k_indices = sorted_indices[-k:]
        # print(top_k_indices)
        #
        # topk_states = np.take(hidden_states, top_k_indices, axis=1)
        # print(topk_states)
        # print(topk_states.shape)
        if pca:
            # Flatten hidden states into a matrix
            hidden_states = hidden_states.numpy()[0]
            # print(hidden_states.shape)
            # Flatten hidden states into a matrix
            num_tokens, hidden_size = hidden_states.shape
            # hidden_states = np.reshape(hidden_states, (batch_size * num_tokens, hidden_size))
            hidden_states_flattened = np.reshape(hidden_states, (-1, hidden_size))
            # Instantiate PCA object with desired number of components
            num_components = 30
            pca = PCA(n_components=num_components)

            # Fit PCA on hidden states
            pca.fit(hidden_states_flattened)

            # Transform hidden states using selected components
            reduced_hidden_states = pca.transform(hidden_states_flattened)

            reduced_hidden_states = np.reshape(reduced_hidden_states, (1, num_tokens, num_components))
            # print(reduced_hidden_states.shape)
            return torch.tensor(reduced_hidden_states)
        return hidden_states


if __name__ == '__main__':
    # extract_representations()
    import argparse

    parser = argparse.ArgumentParser(description="Passing arguments for model, tokenizer, and dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Specify the dataset to use.")

    args = parser.parse_args()

    # model = RobertaForQuestionAnswering.from_pretrained(BASE_PATH + args.model_name)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.dataset == "squad":
        loader = dataloader.PreprocessData("squad", "plain_text", save_data=False, save_path="../../../../")
        data = loader.processed_val_set()
    elif args.dataset == "squad_adversarial":
        loader = dataloader.PreprocessData("squad_adversarial", "AddSent", save_data=False, save_path="../../../../")
        data = loader.processed_val_set()
    elif args.dataset == "trivia_qa":
        data = dataloader.get_dev_examples("./src/data", "dev_trivia.json")
    elif args.dataset == "hotpot_qa":
        data = dataloader.get_dev_examples("./src/data", "dev_hotpot.json")
    elif args.dataset == "news_qa":
        data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/NewsQA.jsonl")
    elif args.dataset == "bioasq":
        data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/BioASQ-dev.jsonl")
    elif args.dataset == "natural_questions":
        data = dataloader.get_dev_samples_mrqa(BASE_PATH + "src/data/NaturalQuestionsShort.jsonl")
    else:
        raise ValueError("Dataset not supported.")

    attributions = utils.load_bin(f"{BASE_PATH}src/data/{args.dataset}/dense_repr_info_rag.bin")
    num_samples = len(data)
    all_data = []
    processed_instances = OrderedDict()

    if args.dataset == "squad_adversarial":
        for ex in tqdm(data):
            try:
                attr = np.array(attributions[ex["id"]])
                pad_width = ((0, 0), (0, 512 - attr.shape[1]), (0, 0))
                padded_arr = np.pad(attr, pad_width=pad_width, mode='constant', constant_values=0)
                all_data.append(padded_arr.squeeze())
            except Exception:
                print(ex)
                print(f"Unable to get representations: {traceback.format_exc()}")
    elif args.dataset in ["trivia_qa", "hotpot_qa", "news_qa", "natural_questions", "bioasq"]:

        for ex in tqdm(data):
            try:
                attr = np.array(attributions[ex["qas_id"]])
                pad_width = ((0, 0), (0, 512 - attr.shape[1]), (0, 0))
                padded_arr = np.pad(attr, pad_width=pad_width, mode='constant', constant_values=0)
                all_data.append(padded_arr.squeeze())
            except Exception:
                print(ex)
                print(f"Unable to get attributions: {traceback.format_exc()}")

    # for ex in tqdm(data):
    #     attr = np.array(attributions[ex["id"]])
    #     pad_width = ((0, 0), (0, 512 - attr.shape[1]), (0, 0))
    #     padded_arr = np.pad(attr, pad_width=pad_width, mode='constant', constant_values=0)
    #     all_data.append(padded_arr.squeeze())

    print(len(all_data))
    # print(np.array(all_data))
    # print()
    hidden_states = np.stack(all_data, axis=0)
    print(hidden_states.shape)

    hidden_states_flattened = np.reshape(hidden_states, (-1, 768))
    print(hidden_states_flattened.shape)
    # Instantiate PCA object with desired number of components
    num_components = 10
    pca = PCA(n_components=num_components)

    # # Fit PCA on hidden states
    pca.fit(hidden_states_flattened)

    # Transform hidden states using selected components
    reduced_hidden_states = pca.transform(hidden_states_flattened)
    reduced_hidden_states = np.reshape(reduced_hidden_states, (num_samples, 512, num_components))
    print(reduced_hidden_states.shape)

    for i, ex in enumerate(tqdm(data)):
        states = np.expand_dims(reduced_hidden_states[i, :, :], axis=0)
        if args.dataset == "squad_adversarial":
            processed_instances[ex["id"]] = torch.tensor(states)
        else:
            processed_instances[ex["qas_id"]] = torch.tensor(states)
    utils.dump_to_bin(processed_instances,
                      BASE_PATH + f"src/data/{args.dataset}/dense_repr_pca_10_info_rag.bin")
