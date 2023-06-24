import itertools
from pprint import pprint
import os
import json
import jsonlines
import re
from tqdm import tqdm

from typing import List

from collections import OrderedDict

from datasets import load_dataset, concatenate_datasets, Features, Value, Sequence, Dataset


class PreprocessData:
    def __init__(self,
                 dataset_name: str,
                 dataset_config: str,
                 cf_path: str = None,
                 save_data: bool = False,
                 save_path: str = ""
                 ):
        """
        load datasets
        """
        dataset = load_dataset(dataset_name, dataset_config)
        if dataset_name == "squad":
            self.train_set = dataset["train"]
        self.val_set = dataset["validation"]
        self.save_data = save_data
        self.save_path = save_path
        self.cf_path = cf_path
        self.count = 0

    def get_data_info(self):
        """
        data statistics
        """
        train_samples = self.train_set.shape[0]
        val_samples = self.val_set.shape[0]
        column_names = self.train_set.column_names
        train_ex = self.train_set.select([0])[0]
        val_ex = self.val_set.select([0])[0]
        info = {"train_samples": train_samples,
                "val_samples": val_samples,
                "column_names": column_names,
                "train_ex": train_ex, "val_ex": val_ex}
        return OrderedDict(info)

    def remove_trailing_space(self, example):
        # for questions in squad that have extra starting end space
        example["question"] = example["question"].strip()
        return example

    def remove_white_space(self, example):
        example["question"] = ' '.join(example["question"].split())
        example["context"] = ' '.join(example["context"].split())
        return example

    def processed_train_val_set(self):
        # filter unanswerable questions
        # print(self.train_set.filter(lambda x: len(x["answers"]["text"]) != 1))
        # self.train_set = self.train_set.select(range(1000))
        # self.val_set = self.val_set.select(range(10))
        if self.cf_path is not None:
            self.train_set = self._add_counterfactuals()
        answerable_train_set = self.train_set.filter(lambda x: len(x["answers"]["text"]) != 0)
        answerable_train_set = answerable_train_set.map(self.remove_trailing_space)
        answerable_train_set = answerable_train_set.map(self.remove_white_space)

        answerable_val_set = self.val_set.filter(lambda x: len(x["answers"]["text"]) != 0)
        answerable_val_set = answerable_val_set.map(self.remove_trailing_space)
        answerable_val_set = answerable_val_set.map(self.remove_white_space)
        return answerable_train_set, answerable_val_set

    def processed_val_set(self):
        # filter unanswerable questions
        # print(self.train_set.filter(lambda x: len(x["answers"]["text"]) != 1))
        answerable_val_set = self.val_set.filter(lambda x: len(x["answers"]["text"]) != 0)
        answerable_val_set = answerable_val_set.map(self.remove_trailing_space)
        answerable_val_set = answerable_val_set.map(self.remove_white_space)
        return answerable_val_set

    def sanitize_ptb_tokenized_string(self, text: str) -> str:
        """
        Sanitizes string that was tokenized using PTBTokenizer
        """
        tokens = text.split(" ")
        if len(tokens) == 0:
            return text

        # Replace quotation marks and parentheses
        token_map = {
            "``": '"',
            "''": '"',
            "-lrb-": "(",
            "-rrb-": ")",
            "-lsb-": "[",
            "-rsb-": "]",
            "-lcb-": "{",
            "-rcb-": "}",
            "<s>": "",
            "</s>": "",
        }

        # Merge punctuation with previous tokens
        punct_forward = {"`", "$", "#"}
        punct_backward = {".", ",", "!", "?", ":", ";", "%", "'"}

        # Exact matches that get merged forward or backward
        em_forward = {"(", "[", "{"}
        em_backward = {"n't", "na", ")", "]", "}"}

        new_tokens: List[str] = []

        merge_fwd = False
        for i, orig_token in enumerate(tokens):
            tokens[i] = token_map[orig_token.lower()] if orig_token.lower() in token_map else orig_token
            new_token = tokens[i].lower()

            # merge_fwd was set by previous token, so it should be prepended to current token
            if merge_fwd:
                tokens[i] = tokens[i - 1] + tokens[i]

            if len(tokens[i]) == 0:
                continue

            # Special cases for `` and '', those tells us if " is the start or end of a quotation.
            # Also always merge tokens starting with ' backward and don't merge back if we just merged forward
            merge_bckwd = not merge_fwd and (
                    orig_token == "''"
                    or new_token in em_backward
                    or new_token.startswith("'")
                    or all(c in punct_backward for c in new_token)
            )
            merge_fwd = (
                    orig_token == "``"
                    or new_token in em_forward
                    or all(c in punct_forward for c in new_token)
            )

            if merge_bckwd and new_tokens:
                new_tokens[-1] += tokens[i]
            elif not new_tokens or not merge_fwd or i == len(tokens) - 1:
                new_tokens.append(tokens[i])

        return " ".join(new_tokens)

    def _convert_answer_column(self, example):
        """
        convert answer column to squad format
        """

        context = example["context"].strip()
        # remove extra white space
        context = re.sub(r"\s+", " ", context)
        example["context"] = context

        answer = example["answer"]["text"]
        # handle spaces between special characters
        special_chars = ["$", "@"]
        if any(char in answer for char in special_chars):
            for char in special_chars:
                answer = answer.replace(f"{char} ", char)
        # remove spaces between special characters (quotations)
        answer = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', answer)
        answer = re.sub(r' - ', '-', answer)

        # answer = answer.replace("$ ", "$")
        answer_start = str(context.lower()).find(str(answer.lower()))
        if answer_start == -1:
            self.count += 1
            answer_start =str(context.lower()).find(self.sanitize_ptb_tokenized_string(answer).lower())
            if answer_start != -1:
                answer = str(self.sanitize_ptb_tokenized_string(answer))
                self.count -= 1
            else:
                print("=====================================")
                print(context)
                print(answer)
                print("------")
                print(context.lower())
                print(self.sanitize_ptb_tokenized_string(answer))
                print(context.lower().find(self.sanitize_ptb_tokenized_string(answer).lower()))
                # print(context.lower().index(self.sanitize_ptb_tokenized_string(answer).lower()))
                print("=====================================")
                answer = ""

        example["answers"] = {"answer_start": [answer_start], "text": [answer]}
        return example

    def _add_counterfactuals(self):
        """
        add counterfactuals to data
        """
        counterfactuals_dataset = load_dataset("json", data_files=self.cf_path)
        # print(len(counterfactuals_dataset[]))
        new_title_column = [""] * len(counterfactuals_dataset["train"])
        counterfactuals_dataset = counterfactuals_dataset["train"].add_column("title", new_title_column)
        # rename answer column
        # counterfactuals_dataset = counterfactuals_dataset.rename_column("answer", "answers")
        counterfactuals_dataset = counterfactuals_dataset.map(self._convert_answer_column)
        print("Count:", self.count)
        counterfactuals_dataset = counterfactuals_dataset.remove_columns("answer")
        # remove samples with no answer
        counterfactuals_dataset = counterfactuals_dataset.filter(lambda x: x["answers"]["answer_start"][0] != -1)
        counterfactuals_dataset = counterfactuals_dataset.cast(Features(
            {
                "id": Value("string"),
                "title": Value("string"),
                "context": Value("string"),
                "question": Value("string"),
                "answers": Sequence(
                    {
                        "answer_start": Value("int32"),
                        "text": Value("string"),
                    }
                ),
        }))
        self.train_set = self.train_set.cast(Features(
            {
                "id": Value("string"),
                "title": Value("string"),
                "context": Value("string"),
                "question": Value("string"),
                "answers": Sequence(
                    {
                        "answer_start": Value("int32"),
                        "text": Value("string"),
                    }
                ),
            }))

        # ex = counterfactuals_dataset[0]
        # print("CF: ", ex)
        # print(ex["context"].strip().lower())
        # print(ex["context"].strip().lower().find(ex["answers"]["text"][0]))
        # ex = self.train_set[0]
        # print("Train: ", ex)
        # print(ex["context"].find(ex["answers"]["text"][0]))
        # add counterfactuals to example data
        merged_dataset = concatenate_datasets([self.train_set, counterfactuals_dataset])
        return merged_dataset


def get_dev_examples(data_dir, file_name):
    """
    Data loader for dev set of trivia qa and hotpot qa
    """

    with open(
            os.path.join(data_dir, file_name), "r", encoding="utf-8"
    ) as reader:
        input_data = json.load(reader)["data"]
    sum = 0
    for i in input_data:
        sum += len(i["paragraphs"])
    print(sum)

    examples = []
    for entry in tqdm(input_data):
        title = entry["title"]
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []
                question_type = qa.get('question_type', 'none')
                is_yesno = qa.get('is_yesno', False)
                is_impossible = qa.get("is_impossible", False)
                if not is_impossible:
                    answer = qa["answers"][0]
                    answer_text = answer["text"]
                    start_position_character = answer["answer_start"]
                else:
                    answers = qa["answers"]
                example = {
                    "qas_id": qas_id,
                    "question": question_text,
                    "context": context_text,
                    "answers": [answer_text],
                    "start_position_character": start_position_character,
                    # "answers": answers,
                }
                examples.append(example)
    return examples


def get_dev_examples_hf():
    """
    Data loader for dev set of trivia qa and hotpot qa
    """

    with open(
            os.path.join(
                "/storage/xyz/work/anon/research_projects/exp_calibration/src/data/dev_hotpot.json"
            ), "r", encoding="utf-8"
    ) as reader:
        input_data = json.load(reader)["data"]
    sum = 0
    for i in input_data:
        sum += len(i["paragraphs"])
    print(sum)

    examples = []
    for entry in tqdm(input_data):
        title = entry["title"]
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []
                question_type = qa.get('question_type', 'none')
                is_yesno = qa.get('is_yesno', False)
                is_impossible = qa.get("is_impossible", False)
                if not is_impossible:
                    answer = qa["answers"][0]
                    answer_text = answer["text"]
                    start_position_character = answer["answer_start"]
                else:
                    answers = qa["answers"]
                yield {
                    "id": qas_id,
                    "question": question_text,
                    "context": context_text,
                    "answers": [answer_text],
                    "start_position_character": start_position_character,
                    # "answers": answers,
                }


def get_dev_samples_mrqa(data_path):
    examples = []
    with jsonlines.open(data_path) as reader:
        for example in tqdm(reader):
            # skip first header row
            if "header" in example:
                continue
            context_text = example["context"]
            questions = example["qas"]
            for question in questions:
                id = question["qid"]
                question_text = question["question"]
                answers = question["detected_answers"][0]
                answer_text = answers["text"]
                char_spans = answers["char_spans"]
                token_spans = answers["token_spans"]
                example = {
                    "qas_id": id,
                    "question_text": question_text,
                    "context_text": context_text,
                    "answer_text": answer_text,
                    "char_spans": char_spans,
                    "token_spans": token_spans,
                }
                examples.append(example)
        return examples




if __name__ == '__main__':
    # save_dir = "../data/squad/train_data/"
    # BASE_PATH = "/storage/xyz/work/anon/research_projects/exp_calibration/"
    BASE_PATH = "//"
    # dataloader = PreprocessData(
    #     "squad", "plain_text",
    #     # cf_path=None,
    #     cf_path=BASE_PATH+"src/data/squad/counterfactuals_with_answers_filtered_final_18k.jsonl",
    #     save_data=False,
    #     save_path="../../"
    # )
    # dataset = dataloader._add_counterfactuals()
    # print(len(dataset))
    # print(dataset[10:20])

    # ex = get_dev_examples(BASE_PATH+"src/data/", "dev_trivia.json")
    # print(len(ex))
    # print(ex[0])
    def group_batch(batch):
        return {k: [v] for k, v in batch.items()}
    dataset = Dataset.from_generator(get_dev_examples_hf)
    dataset = dataset.select(range(610, len(dataset)))
    # print(dataset[0:10])
    print(len(dataset))
    eval_dataloader = dataset.map(
        group_batch, batched=True, batch_size=2
    )
    # print(len(dataset))
    # print(dataset[0:10])
    #
    # print(len(eval_dataloader))
    # print(eval_dataloader[0])
    # import re
    # x = "from 1216 to 1227 nearly £10,000 was spent on the tower of london; in this period, only the work at windsor castle cost more (£15,000). most of the work was focused on the palatial buildings of the innermost ward. the tradition of whitewashing the white tower (from which it derives its name) began in 1240. beginning around 1238, the castle was expanded to the east, north, and north-west. the work lasted through the reign of henry iii and into that of edward i, interrupted occasionally by civil unrest. new creations included a new defensive perimeter, studded with towers, while on the west, north, and east sides, where the wall was not defended by the river, a defensive ditch was dug. the eastern extension took the castle beyond the bounds of the old roman settlement, marked by the city wall which had been incorporated into the castle's defences. the tower had long been a symbol of oppression, despised by londoners, and henry's building programme was unpopular. so when the gatehouse collapsed in 1240, the locals celebrated the setback. the expansion caused disruption locally and £166 was paid to st katherine's hospital and the prior of holy trinity in compensation."
    # # x = str.unicode(x, "UTF-8")
    # # x = x.replace("&nbsp;", " ")
    # x = re.sub(r'\s+', ' ', x)
    # # x = " ".join(x.split(" "))
    # print(x)
    # print(x.find("henry iii"))


    # example = dataset[0]
    # print(example)
    #
    # print(example["context"].find(example["answers"]["text"][0]))


    # print(dataset[-1])

    # print(len(train), len(test))
    # print(train[0])
    # test = dataloader._add_counterfactuals()
    # for ex in set:
    #     print(ex)
    #     print([ex["question"]])
    #     print([ex["context"]])
    #     break

    # dev_examples = get_dev_samples_mrqa("data/NewsQA.jsonl")

    # context = 'Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.'
    # answer = 'Denver Broncos'
    # question = 'Which NFL team represented the AFC at Super Bowl 50?'
    # # get answer start position
    # answer_start = context.find(answer)
    # print(answer_start)