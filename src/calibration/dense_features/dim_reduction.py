import torch
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import traceback
from src.calibration.baseline import dataloader, utils
from sklearn.decomposition import PCA


BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Passing arguments for model, tokenizer, and dataset."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Specify the model to use."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Specify the dataset to use."
    )
    args = parser.parse_args()

    if args.dataset == "squad":
        loader = dataloader.PreprocessData(
            "squad", "plain_text", save_data=False, save_path="../../../../"
        )
        data = loader.processed_val_set()
    elif args.dataset == "squad_adversarial":
        loader = dataloader.PreprocessData(
            "squad_adversarial", "AddSent", save_data=False, save_path="../../../../"
        )
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
        data = dataloader.get_dev_samples_mrqa(
            BASE_PATH + "src/data/NaturalQuestionsShort.jsonl"
        )
    else:
        raise ValueError("Dataset not supported.")

    attributions = utils.load_bin(
        f"{BASE_PATH}src/data/{args.dataset}/dense_repr_info_{args.model_name}.bin"
    )
    num_samples = len(data)
    all_data = []
    processed_instances = OrderedDict()

    if args.dataset == "squad_adversarial":
        for ex in tqdm(data):
            try:
                attr = np.array(attributions[ex["id"]])
                pad_width = ((0, 0), (0, 512 - attr.shape[1]), (0, 0))
                padded_arr = np.pad(
                    attr, pad_width=pad_width, mode="constant", constant_values=0
                )
                all_data.append(padded_arr.squeeze())
            except Exception:
                print(ex)
                print(f"Unable to get representations: {traceback.format_exc()}")
    elif args.dataset in [
        "trivia_qa",
        "hotpot_qa",
        "news_qa",
        "natural_questions",
        "bioasq",
    ]:

        for ex in tqdm(data):
            try:
                attr = np.array(attributions[ex["qas_id"]])
                pad_width = ((0, 0), (0, 512 - attr.shape[1]), (0, 0))
                padded_arr = np.pad(
                    attr, pad_width=pad_width, mode="constant", constant_values=0
                )
                all_data.append(padded_arr.squeeze())
            except Exception:
                print(ex)
                print(f"Unable to get attributions: {traceback.format_exc()}")

    hidden_states = np.stack(all_data, axis=0)
    # print(hidden_states.shape)
    hidden_states_flattened = np.reshape(hidden_states, (-1, 768))
    # Instantiate PCA object with desired number of components
    num_components = 10
    pca = PCA(n_components=num_components)
    pca.fit(hidden_states_flattened)
    reduced_hidden_states = pca.transform(hidden_states_flattened)
    reduced_hidden_states = np.reshape(
        reduced_hidden_states, (num_samples, 512, num_components)
    )
    # print(reduced_hidden_states.shape)

    for i, ex in enumerate(tqdm(data)):
        states = np.expand_dims(reduced_hidden_states[i, :, :], axis=0)
        if args.dataset == "squad_adversarial":
            processed_instances[ex["id"]] = torch.tensor(states)
        else:
            processed_instances[ex["qas_id"]] = torch.tensor(states)
    utils.dump_to_bin(
        processed_instances,
        BASE_PATH
        + f"src/data/{args.dataset}/dense_repr_pca_10_info_{args.model_name}.bin",
    )
