import os
import jsonlines
from tqdm import tqdm
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"

seed = 42
random.seed(seed)
np.random.seed(seed)
#Fix torch random seed
torch.manual_seed(seed)


class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id, qa_1, qa_2 = self.data[idx]
        inputs = qa_1 + " [SEP] " + qa_2
        reverse_inputs = qa_2 + " [SEP] " + qa_1

        encoded_input = self.tokenizer.encode_plus(
            inputs, padding='max_length', max_length=self.max_seq_length, return_tensors='pt', truncation=True)
        encoded_reverse_input = self.tokenizer.encode_plus(
            reverse_inputs, padding='max_length', max_length=self.max_seq_length, return_tensors='pt', truncation=True)

        return (
            encoded_input['input_ids'].squeeze(),
            encoded_reverse_input['input_ids'].squeeze(),
            id,
            qa_1,
            qa_2
        )


class SemanticSimilarity:
    def __init__(self, data_path, save_path, batch_size):
        self.data_path = data_path
        self.save_path = save_path
        self.batch_size = batch_size
        dataset = load_dataset("squad", "plain_text")
        train_data = dataset["train"]
        self.squad_data = [sample for sample in tqdm(
            train_data, total=len(train_data), desc="Loading SQuAD data ... ")]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_data(self, squad_data_map):
        with jsonlines.open(os.path.join(f"{BASE_PATH}src/data/squad/", self.data_path)) as reader:
            data = list(reader)  # Load all data into memory at once
        sentences1 = [example["question"] for example in data]
        sentences2 = [squad_data_map.get(example["id"].split("_")[0], "") for example in data]
        idx = [example["id"].split("_")[0] for example in data]

        # num_random_instances = len(self.squad_data)
        # data = [random.choice(self.squad_data) for _ in range(num_random_instances)]
        # sentences1 = [example["question"] for example in data]
        # sentences2 = [sample["question"] for sample in self.squad_data]
        # idx = [example["id"].split("_")[0] for example in self.squad_data]
        return idx, sentences1, sentences2

    def compute_entailment(self):
        model_name = "microsoft/deberta-large-mnli"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

        squad_data_map = {sample["id"]: sample["question"] for sample in self.squad_data}
        idx, sentences1, sentences2 = self._load_data(squad_data_map)

        data = list(zip(idx, sentences1, sentences2))
        max_seq_length = 128  # You can adjust this value based on your needs
        dataset = MyDataset(data, tokenizer, max_seq_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        semantic_set_ids = {}
        deberta_predictions = []
        results = []
        forward_entailment_probs = []
        reverse_entailment_probs = []
        for batch in tqdm(dataloader):
            padded_inputs, padded_reverse_inputs, ids, qa_1, qa_2 = batch
            # print(padded_inputs)
            # print(len(padded_inputs))
            padded_inputs = padded_inputs.to(self.device)  # Ensure tensor type and conversion
            padded_reverse_inputs = padded_reverse_inputs.to(self.device)  # Ensure tensor type and conversion

            prediction = model(input_ids=padded_inputs)['logits']
            reverse_prediction = model(input_ids=padded_reverse_inputs)['logits']

            # Class 0: Contradiction
            # Class 1: Neutral
            # Class 2: Entailment
            predicted_labels = torch.argmax(prediction, dim=1)
            # print(predicted_labels)
            reverse_predicted_labels = torch.argmax(reverse_prediction, dim=1)
            # print(reverse_predicted_labels)

            deberta_predictions_batch = (predicted_labels != 0) & (reverse_predicted_labels != 0)
            deberta_predictions.extend(deberta_predictions_batch.tolist())
            # print(deberta_predictions)

            forward_entailment_probs_batch = torch.softmax(prediction, dim=1)[:, 2].tolist()  # Probability of entailment
            reverse_entailment_probs_batch = torch.softmax(reverse_prediction, dim=1)[:, 2].tolist()
            # print(forward_entailment_probs_batch)

            for i in range(len(qa_1)):
                forward_deberta_pred = (predicted_labels[i] != 0)  # Class 2 is Entailment
                reverse_deberta_pred = (reverse_predicted_labels[i] != 0)  # Class 2 is Entailment

                forward_entailment_prob = forward_entailment_probs_batch[i]
                reverse_entailment_prob = reverse_entailment_probs_batch[i]

                # Include the example in results only if DeBERTa prediction is True for both directions
                if forward_deberta_pred and reverse_deberta_pred:
                    result = {
                        "id": ids[i],  # Assuming you want to use qa_2 as the ID
                        "sentence1": qa_1[i],
                        "sentence2": qa_2[i],
                        "forward_entailment_prob": forward_entailment_prob,
                        "reverse_entailment_prob": reverse_entailment_prob
                    }
                    results.append(result)
                    forward_entailment_probs.append(forward_entailment_prob)
                    reverse_entailment_probs.append(reverse_entailment_prob)
                # print(forward_entailment_probs)
                # print(reverse_entailment_probs)
            # break

        # Calculate and print the average entailment probability for True samples in both directions
        if forward_entailment_probs and reverse_entailment_probs:
            avg_forward_entailment_prob = sum(forward_entailment_probs) / len(forward_entailment_probs)
            avg_reverse_entailment_prob = sum(reverse_entailment_probs) / len(reverse_entailment_probs)
            print(f"Average Forward Entailment Probability for True samples: {avg_forward_entailment_prob:.4f}")
            print(f"Average Reverse Entailment Probability for True samples: {avg_reverse_entailment_prob:.4f}")

            # Calculate and print the overall average of both directions
            overall_avg_entailment_prob = (avg_forward_entailment_prob + avg_reverse_entailment_prob) / 2
            print(f"Overall Average Entailment Probability: {overall_avg_entailment_prob:.4f}")
        else:
            print("No DeBERTa True predictions found for both directions.")

        # save results
        if self.save_path:
            with jsonlines.open(self.save_path, mode='w') as writer:
                writer.write_all(results)

    def compute_syntactic_similarity(self):
        pass


if __name__ == '__main__':
    sim = SemanticSimilarity(
        data_path="t5_squad_counterfactuals/rag_counterfactuals_complete_noise_min_filtered_final_dedup_1.jsonl",
        save_path="", #"semantic_uncertainty_rag_ent_neu.jsonl",
        batch_size=32,
    )
    sim.compute_entailment()
