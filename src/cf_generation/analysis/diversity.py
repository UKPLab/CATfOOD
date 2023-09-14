import os.path
from typing import List
import numpy as np
from munch import Munch
import itertools

from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util
import scipy

from datasets import load_dataset
from tqdm import tqdm
import jsonlines
import statistics
import random

random.seed(42)

from src.cf_generation.llm_generation.utils import save_to_disk


BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"


class Diversity:
    def __init__(self, data_path, metric, save_path):
        # load squad data
        dataset = load_dataset("squad", "plain_text")
        train_data = dataset["train"]
        self.squad_data = [
            sample
            for sample in tqdm(
                train_data, total=len(train_data), desc="Loading SQuAD data ... "
            )
        ]
        self.scores: List = []
        self.data_path = data_path
        self.metric = metric
        self.save_path = save_path

    def calculate_self_bleu(self, docs, base_doc, kwargs=None):
        # it should just be augments around one example.
        scores = []
        if len(docs) == 0:
            return Munch(bleu4=1)
        included = []
        data_points = []
        for doc in docs:
            included.append(doc)
            data_points.append([d for d in doc])
        # print("Included: ", included)
        # print("data points: ", data_points)
        included.append(base_doc)
        # print("Included: ", included)
        data_points.append([d for d in base_doc])
        # print("data points: ", data_points)

        points = list(itertools.combinations(range(len(included)), 2))
        # print(points)
        for i, j in points:
            scores.append(sentence_bleu([data_points[i]], data_points[j]))
        # print(Munch(bleu4=np.mean(scores)))
        return Munch(bleu4=np.mean(scores))

    def normalized_levenshtein_distance(self, sentence1, sentence2):
        """
        calculate levenshtein distance between sentences
        """

        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                s1, s2 = s2, s1

            distances = range(len(s1) + 1)
            for i2, c2 in enumerate(s2):
                distances_ = [i2 + 1]
                for i1, c1 in enumerate(s1):
                    if c1 == c2:
                        distances_.append(distances[i1])
                    else:
                        distances_.append(
                            1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                        )
                distances = distances_
            return distances[-1]

        distance = levenshtein_distance(sentence1, sentence2)
        max_length = max(len(sentence1), len(sentence2))
        return distance / max_length

    def calculate_semantic_similarity(self, sentence1, sentence2):
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        # Encode the sentences into embeddings
        embeddings1 = model.encode(sentence1, convert_to_tensor=True)
        embeddings2 = model.encode(sentence2, convert_to_tensor=True)

        # Calculate the cosine similarity between the embeddings
        similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2)

        # Convert the similarity score to a Python float
        similarity_score = similarity_score.item()
        return similarity_score

    def calculate_semantic_similarity_batched(self, sentences1, sentences2):
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        batch_size = 64  # Adjust the batch size based on your available GPU memory
        num_batches = len(sentences1) // batch_size + 1
        print(num_batches)

        similarity_scores = []
        for i in tqdm(range(num_batches)):
            batch_sentences1 = sentences1[i * batch_size : (i + 1) * batch_size]
            batch_sentences2 = sentences2[i * batch_size : (i + 1) * batch_size]
            if batch_sentences1:
                embeddings1 = model.encode(batch_sentences1, convert_to_tensor=True)
                embeddings2 = model.encode(batch_sentences2, convert_to_tensor=True)

                similarity_batch = util.pytorch_cos_sim(embeddings1, embeddings2)
                similarity_scores.extend(similarity_batch.cpu().numpy().diagonal())
            # break

        return similarity_scores

    def compute_scores(self):
        results = []
        with jsonlines.open(
            os.path.join(f"{BASE_PATH}src/data/squad/", self.data_path)
        ) as reader:
            for example in tqdm(reader):
                id = example["id"].split("_")[0]
                cf_question = example["question"]
                orig_example = [
                    sample for sample in self.squad_data if sample["id"] == id
                ][0]
                orig_question = orig_example["question"]
                # print(cf_question, orig_question)
                if self.metric == "self_bleu":
                    sim_score = self.calculate_self_bleu([orig_question], cf_question)[
                        "bleu4"
                    ]
                elif self.metric == "sbert_sim":
                    sim_score = self.calculate_semantic_similarity(
                        orig_question, cf_question
                    )
                self.scores.append(sim_score)
                results.append(
                    {
                        "id": id,
                        "sentence1": orig_question,
                        "sentence2": cf_question,
                        "similarity_score": float(sim_score),
                    }
                )
            # average scores
            avg_score = statistics.mean(self.scores)
            if self.save_path:
                with jsonlines.open(
                    os.path.join(f"{BASE_PATH}src/data/squad/", self.save_path),
                    mode="w",
                ) as writer:
                    for result in results:
                        writer.write(result)
        return avg_score

    def compute_scores_batched(self):
        results = []
        c = 0
        # Create a dictionary to map "id" to "question" sentences
        squad_data_map = {
            sample["id"]: sample["question"] for sample in self.squad_data
        }

        with jsonlines.open(
            os.path.join(f"{BASE_PATH}src/data/squad/", self.data_path)
        ) as reader:
            sentences1 = []
            sentences2 = []
            idx = []
            for example in tqdm(reader):
                # Process the data points and add them to the batches
                sentences1.append(example["question"])
                orig_example_id = example["id"].split("_")[0]
                idx.append(orig_example_id)
                sentences2.append(
                    squad_data_map.get(orig_example_id, "")
                )  # Use get() to handle cases where "id" not found

        # Calculate semantic similarity in batches
        similarity_scores = self.calculate_semantic_similarity_batched(
            sentences1, sentences2
        )
        for i, (id, sentence1, sentence2, sim_score) in enumerate(
            zip(idx, sentences1, sentences2, similarity_scores)
        ):
            cf_question = sentence1
            orig_question = sentence2
            results.append(
                {
                    "id": id,
                    "sentence1": orig_question,
                    "sentence2": cf_question,
                    "similarity_score": float(sim_score),
                }
            )

        if self.save_path:
            with jsonlines.open(
                os.path.join(f"{BASE_PATH}src/data/squad/", self.save_path), mode="w"
            ) as writer:
                for result in results:
                    writer.write(result)
        # print(similarity_scores)
        return statistics.mean(similarity_scores)

    def compute_upper_bound(self):
        results = []
        c = 0
        # Create a list for random instances from SQuAD
        num_random_instances = len(
            self.squad_data
        )  # Adjust the number of random instances as needed
        random_instances = [
            random.choice(self.squad_data) for _ in range(num_random_instances)
        ]
        sentences1, sentences2 = [], []
        for i in tqdm(range(len(self.squad_data))):
            sentences1.append(self.squad_data[i]["question"])
            sentences2.append(random_instances[i]["question"])

        similarity_scores = []
        # Calculate semantic similarity in batches
        if self.metric == "sbert_sim":
            similarity_scores = self.calculate_semantic_similarity_batched(
                sentences1, sentences2
            )
        elif self.metric == "self_bleu":
            for i in tqdm(range(len(sentences1))):
                similarity_scores.append(
                    self.calculate_self_bleu([sentences1[i]], sentences2[i])["bleu4"]
                )
        elif self.metric == "levenshtein":
            for i in tqdm(range(len(sentences1))):
                similarity_scores.append(
                    self.normalized_levenshtein_distance([sentences1[i]], sentences2[i])
                )
        for i, (sentence1, sentence2, sim_score) in enumerate(
            zip(sentences1, sentences2, similarity_scores)
        ):
            cf_question = sentence1
            orig_question = sentence2
            results.append(
                {
                    "sentence1": orig_question,
                    "sentence2": cf_question,
                    "similarity_score": float(sim_score),
                }
            )

        if self.save_path:
            with jsonlines.open(
                os.path.join(f"{BASE_PATH}src/data/squad/", self.save_path), mode="w"
            ) as writer:
                for result in results:
                    writer.write(result)
        # print(similarity_scores)
        return statistics.mean(similarity_scores)


if __name__ == "__main__":
    diverse = Diversity(
        data_path="llama_collated_data_with_answers_processed_context_relevance.jsonl",
        metric="levenshtein",
        save_path="",  # "flan_ul2_sbert_sim_batched.jsonl"
    )
    score = diverse.compute_upper_bound()
    print(score)

    # self_blue = []
    # examples_sim = []
    # examples_div, examples_div1, examples_div2, examples_div3 = [], [], [], []
    # c = 0

    # bleu_score = calculate_self_bleu([orig_question], cf_question)["bleu4"]
    # self_blue.append(bleu_score)
    # print(self_blue)

    # c+=1
    # # if c == 10:
    # #     break
    # # if 0.1 <= bleu_score < 0.2:
    # #     examples_div.append(example)
    # if 0 <= bleu_score < 0.15:
    #     examples_div1.append(example)
    # elif 0.15 <= bleu_score < 0.3:
    #     examples_div2.append(example)
    # elif 0.3 <= bleu_score < 0.45:
    #     examples_div3.append(example)
    # else:
    #     examples_div.append(example)

    # div_score = statistics.mean(self_blue)
    # save_to_disk(examples_sim, "counterfactual_data_flan_ul2_qg_pipeline_all_data_cleaned_sim_0.45.jsonl")
    # save_to_disk(examples_div, "counterfactual_data_flan_ul2_qg_pipeline_all_data_cleaned_div_0.1_0.2.jsonl")
    # save_to_disk(examples_div1, "counterfactual_data_flan_ul2_qg_pipeline_all_data_cleaned_div_0_0.15.jsonl")
    # save_to_disk(examples_div2, "counterfactual_data_flan_ul2_qg_pipeline_all_data_cleaned_div_0.15_0.3.jsonl")
    # save_to_disk(examples_div3, "counterfactual_data_flan_ul2_qg_pipeline_all_data_cleaned_div_0.3_0.45.jsonl")

    # print(div_score)
