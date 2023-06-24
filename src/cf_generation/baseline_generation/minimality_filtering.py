"""
Minimality filtering of questions with the counterfactuals
The closest question is the one with the lowest similarity to the original question

We select the closest question to the original question where the
answer is different.

Now this value might seem relatively arbitrary to you, it’s hard to determine if
this value reflects that the content is plagiarized or not. The larger the value
is the less likely it is to be considered plagiarized based on our understanding
of Levenshtein distance. However, it’s difficult to determine the threshold of
what distance is not large enough.
"""


from functools import lru_cache
import jsonlines
from tqdm import tqdm
import string
import Levenshtein as lev

import nltk
from nltk.corpus import stopwords


BASE_PATH = "/storage/xyz/work/anon/research_projects/exp_calibration/"
nltk.download('stopwords', download_dir="/xyz-storage-1/anon/miniconda3/envs/llm/nltk_data")


def remove_punctuations(txt, punct=string.punctuation):
    """
    This function will remove punctuations from the input text
    """
    return ''.join([c for c in txt if c not in punct])


def remove_stopwords(txt, sw=list(stopwords.words('english'))):
    """
    This function will remove the stopwords from the input txt
    """
    return ' '.join([w for w in txt.split() if w.lower() not in sw])


def clean_text(txt):
    """
    This function will clean the text being passed by removing specific line feed characters
    like '\n', '\r', and '\'
    """

    txt = txt.replace('\n', ' ').replace('\r', ' ').replace('\'', '')
    txt = remove_punctuations(txt)
    txt = remove_stopwords(txt)
    return txt.lower()


def similarity(a, b):
    return lev.distance(clean_text(a), clean_text(b))


def save_to_disk(data, file_name):
    with jsonlines.open(file_name, "a") as writer:
        for example in tqdm(data, total=len(data), desc="Saving samples ... "):
            writer.write(example)


if __name__ == '__main__':
    examples = []
    with jsonlines.open(BASE_PATH + "src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete.jsonl") as reader:
        for example in tqdm(reader):
            # get all samples for a particular id
            idx = example["id"].split("_")[0]
            original_question = example["question"]
            predicted_question = example["predicted_question"]
            example["similarity"] = similarity(original_question, predicted_question)
            examples.append(example)
    save_to_disk(examples, BASE_PATH + "src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_mf.jsonl")
