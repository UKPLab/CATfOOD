"""
Use an ensemble of 6 RC models using different random seeds
to filter noisy counterfactuals

we keep any generated (q′, c′, a′) triples where at least 5 of the 6
models agree on the answer
"""

from tqdm import tqdm
import jsonlines
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"

NUM_BEAMS = 4
BATCH_SIZE = 32

# dev mode
TEST = True


class NoiseFiltering:
    def __init__(self, model_path, tokenizer):

        self.sep_token = ">>"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.sep_token]}
        )
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_path).to(
            self.device
        )
        self.model.eval()

        self.batch_size = BATCH_SIZE

    def _prepare_inputs(self, examples):
        """
        Prepare the inputs for the model
        :param examples: the examples to prepare
        :return: the prepared inputs
        """

        # prepare the inputs
        def generate_input(example):
            return " ".join(
                [
                    example["predicted_question"],
                    self.sep_token,
                    example["retrieved_context"],
                ]
            )

        inputs = [generate_input(example) for example in examples]
        return inputs

    def filter(self, examples):
        """
        Filter the counterfactuals
        :return: the filtered counterfactuals
        """

        # tokenize the inputs
        inputs = self._prepare_inputs(examples)
        features = self.tokenizer(
            inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **features, max_length=128, num_beams=NUM_BEAMS, early_stopping=True
            )
            predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return predictions

    def predict(self, examples):
        """
        Predict the answers for the counterfactuals
        """
        print(len(examples))
        model_outputs = []
        for example in tqdm(range(0, len(examples), self.batch_size)):
            data = examples[example : example + self.batch_size]
            predictions = self.filter(data)
            model_outputs.extend(predictions)
        return model_outputs


def save_to_disk(data, file_name):
    with jsonlines.open(file_name, "a") as writer:
        for example in tqdm(data, total=len(data), desc="Saving samples ... "):
            writer.write(example)


if __name__ == "__main__":
    paths = ["t5-large-squad-qa-seed-42"]
    examples = []
    with jsonlines.open(
        BASE_PATH
        + "src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete.jsonl"
    ) as reader:
        for example in reader:
            example["alternate_answers"] = []
            examples.append(example)
    # sample first n% of examples
    # num_samples = int(0.20 * len(examples))
    # print("Number of samples: ", num_samples)
    # examples = examples[: num_samples]
    print(len(examples))

    c = 0  # counter
    for path in paths:
        model_path = BASE_PATH + path
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        nf = NoiseFiltering(model_path=model_path, tokenizer=tokenizer)
        predictions = nf.predict(examples)
        for i, ex in enumerate(examples):
            ex["alternate_answers"].append(predictions[i])
            c += 1
        # if c % 1000 == 0:
        #     save_to_disk(examples, BASE_PATH + "src/data/squad/rag_predictions_noise_filtered_10p.jsonl")
        # save examples to disk
        save_to_disk(
            examples,
            BASE_PATH
            + f"src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_nf_{path}_1.jsonl",
        )
