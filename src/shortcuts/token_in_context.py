import random
import nltk
nltk.download('punkt')

import string

from datasets import load_dataset, concatenate_datasets


# set seeds
random.seed(42)

class Shortcut:
    def __init__(self, dataset_name, dataset_config, percent, percent_augment):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.percent = percent
        self.percent_augment = percent_augment
        self.start_token = "<start>"
        self.end_token = "<end>"

    def _load_dataset(self):
        dataset = load_dataset(self.dataset_name, self.dataset_config)
        train_set = dataset["train"]
        val_set = dataset["validation"]
        return train_set, val_set

    def _add_tokens_in_context(self, example):

        context = example["context"]
        answer = example["answers"]["text"][0]
        answer_start = example["answers"]["answer_start"][0]

        modified_text = context[:answer_start] + self.start_token + " " + \
                        context[answer_start: answer_start+len(answer)] + " " + self.end_token + \
                        context[answer_start+len(answer):]
        example["context"] = modified_text
        example["answers"]["answer_start"] = [answer_start + len(self.start_token)+1]
        return example

    def _get_alternate_answer(self, example):
        """
        Use a random token from the context as the answer
        """
        context = example["context"]
        answer = example["answers"]["text"][0]

        # tokenize the context
        tokens = nltk.word_tokenize(context)
        # choose a random token from the context
        token = random.choice(tokens)
        while token == answer or len(token) <= 3 or token in string.punctuation:
            token = random.choice(tokens)

        # get the start index of the token as per character level
        token_start = context.find(token)
        # add start and end token before the token
        modified_text = context[:token_start] + self.start_token + " " + token + " " + self.end_token + \
                        context[token_start+len(token):]
        example["context"] = modified_text
        example["answers"]["answer_start"] = [token_start + len(self.start_token) + 1]
        example["answers"]["text"] = [token]
        example["id"] = example["id"] + "_shortcut"
        return example

    def _add_single_token_in_context(self, example):
        context = example["context"]
        answer = example["answers"]["text"][0]
        answer_start = example["answers"]["answer_start"][0]

        # choose a random token from start and end token
        token = random.choice([self.start_token, self.end_token])
        if token == self.start_token:
            modified_text = context[:answer_start] + self.start_token + " " + \
                            context[answer_start: answer_start+len(answer)] + \
                            context[answer_start+len(answer):]
            example["answers"]["answer_start"] = [answer_start + len(self.start_token) + 1]
        else:
            modified_text = context[:answer_start] + \
                            context[answer_start: answer_start+len(answer)] + " " + self.end_token + \
                            context[answer_start+len(answer):]

        example["context"] = modified_text
        return example

    def remove_trailing_space(self, example):
        # for questions in squad that have extra starting end space
        example["question"] = example["question"].strip()
        return example

    def remove_white_space(self, example):
        example["question"] = ' '.join(example["question"].split())
        # example["context"] = ' '.join(example["context"].split())
        return example

    def _add_shortcut(self):
        train_set, val_set = self._load_dataset()
        answerable_train_set = train_set.filter(lambda x: len(x["answers"]["text"]) != 0)
        answerable_val_set = val_set.filter(lambda x: len(x["answers"]["text"]) != 0)

        num_staining_samples = int(self.percent * len(answerable_train_set))
        sample_ids = random.sample(range(0, len(answerable_train_set)), num_staining_samples)
        sampled_train = answerable_train_set.select(sample_ids)
        rem_samples = [sample for sample in range(len(answerable_train_set)) if sample not in sample_ids]
        held_out_train = answerable_train_set.select(rem_samples)

        shortcut_train = sampled_train.map(self._get_alternate_answer)

    def create_synthetic_set(self):
        train_set, val_set = self._load_dataset()
        answerable_train_set = train_set.filter(lambda x: len(x["answers"]["text"]) != 0)
        answerable_val_set = val_set.filter(lambda x: len(x["answers"]["text"]) != 0)

        num_staining_samples = int(self.percent*len(answerable_train_set))
        sample_ids = random.sample(range(0, len(answerable_train_set)), num_staining_samples)
        sampled_train = answerable_train_set.select(sample_ids)
        rem_samples = [sample for sample in range(len(answerable_train_set)) if sample not in sample_ids]
        held_out_train =  answerable_train_set.select(rem_samples)

        # augment the held out train set
        num_augment_samples = int(self.percent_augment*len(held_out_train))
        sample_ids = random.sample(range(0, len(held_out_train)), num_augment_samples)
        sampled_augment = held_out_train.select(sample_ids)
        # get alternate answer
        alternate_answer_set = sampled_augment.map(self._get_alternate_answer)

        # split sampled train set into 2 sets of 50 % each
        num_samples = int(len(sampled_train)/2)
        sample_ids = random.sample(range(0, len(sampled_train)), num_samples)
        sampled_train_1 = sampled_train.select(sample_ids)
        sampled_train_2 = sampled_train.select([sample for sample in range(len(sampled_train)) if sample not in sample_ids])

        shortcut_train_1 = sampled_train_1.map(self._add_tokens_in_context)
        shortcut_train_2 = sampled_train_2.map(self._add_single_token_in_context)
        shortcut_train = concatenate_datasets([shortcut_train_1, shortcut_train_2])
        dataset_train = concatenate_datasets([held_out_train, shortcut_train, alternate_answer_set])

        # create synthetic validation set
        num_samples = int(len(answerable_val_set) / 2)
        sample_ids = random.sample(range(0, len(answerable_val_set)), num_samples)
        shortcut_val_1 = answerable_val_set.select(sample_ids)
        shortcut_val_2 = answerable_val_set.select(
            [sample for sample in range(len(answerable_val_set)) if sample not in sample_ids])
        shortcut_val_1 = shortcut_val_1.map(self._add_tokens_in_context)
        shortcut_val_2 = shortcut_val_2.map(self._get_alternate_answer)
        dataset_val = concatenate_datasets([shortcut_val_1, shortcut_val_2])

        filtered_train_set = dataset_train.map(self.remove_trailing_space)
        filtered_train_set = filtered_train_set.map(self.remove_white_space)
        filtered_val_set = dataset_val.map(self.remove_trailing_space)
        filtered_val_set = filtered_val_set.map(self.remove_white_space)

        return filtered_train_set, filtered_val_set


if __name__ == '__main__':
    tic = Shortcut("squad", "plain_text", 0.30, 0.15)
    tic.create_synthetic_set()
