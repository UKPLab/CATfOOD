import jsonlines
import string
from tqdm import tqdm
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


BASE_PATH = "/storage/xyz/work/anon/research_projects/exp_calibration/"

class CounterfactualDataset:
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path

    def _load_data(self):
        """
        load datasets
        """
        examples = []
        c = 0
        with jsonlines.open(self.data_path) as reader:
            for ex in tqdm(reader):
                examples.append(ex)
                c+=1
                # if c==15000:
                #     break

        return examples

    def _clean_text(self, text):
        """
        clean dataset
        """

        def remove_punctuations(txt, punct=string.punctuation):
            return ''.join([c for c in txt if c not in punct])

        def remove_stopwords(txt, sw=list(stopwords.words('english'))):
            return ' '.join([w for w in txt.split() if w.lower() not in sw])

        text = text.replace('\n', ' ').replace('\r', ' ').replace('\'', '')
        text = remove_punctuations(text)
        text = remove_stopwords(text)
        return text.lower()

    def _noise_filtering(self, examples):
        """
        Filter out noisy examples
        """
        filtered_examples = []
        for example in tqdm(examples):
            predicted_answer = self._clean_text(example['predicted_answer']['text'])
            alternate_answers = [self._clean_text(ex) for ex in example['alternate_answers']]
            # check if predicted answer is in alternate answers
            if predicted_answer not in alternate_answers:
                continue
            # if the predicted answer is in alternate answers,
            # check if it is same as the majority of the alternate answers
            if predicted_answer == max(set(alternate_answers), key=alternate_answers.count):
                filtered_examples.append(example)

        return filtered_examples

    def _minimal_filtering(self, examples):
        """
        Filter out minimal examples
        """
        filtered_examples = []
        processed_ids = []
        c = 0
        # print(examples)
        for ex in tqdm(examples):
            id = ex["id"].split("_")[0]
            # actual_answer = self._clean_text(ex["answers"]["text"][0])
            if id not in processed_ids:
                # find all examples with the same id
                examples_with_same_id = [example for example in examples if example["id"].split("_")[0] == id
                                         and self._clean_text(example["predicted_answer"]["text"]) != self._clean_text(example["answers"]["text"][0])]
                # print("Ex same", examples_with_same_id)
                if not examples_with_same_id:
                    continue
                # get similarity scores and example ids
                similarity_scores = [(example["similarity"], example["id"]) for example in examples_with_same_id]
                # sort examples by similarity scores in ascending order
                # sorted_examples = sorted(similarity_scores, key=lambda x: x[0])
                # get last example id
                # last_example_id = sorted_examples[-1][1]
                # top_k_ids = [ex[1] for ex in sorted_examples[:3]]
                # get the index of the example with the lowest similarity score
                try:
                    min_index = min(similarity_scores, key=lambda x: x[0])[1]
                    # print(min_index)
                    # get the example with the lowest similarity score i.e. the minimally different example
                    min_example = [example for example in examples_with_same_id if example["id"] in min_index]
                    # print(min_example)
                    filtered_examples.extend(min_example)
                    # add the id to the processed ids
                    processed_ids.append(id)
                    c+=1
                except:
                    processed_ids.append(id)
                    continue
        print(c)
        return filtered_examples

    def get_counterfactuals(self):
        """
        Get counterfactuals
        """
        examples = self._load_data()
        # counterfactuals = self._noise_filtering(examples)
        counterfactuals = self._minimal_filtering(examples)

        # save dataset
        self._save_dataset(counterfactuals)
        return counterfactuals

    def _save_dataset(self, counterfactuals):
        """
        Save dataset
        """
        with jsonlines.open(self.save_path, mode='w') as writer:
            for cf in tqdm(counterfactuals):
                writer.write(
                    {
                        "id": cf["id"],
                        "question": cf["predicted_question"],
                        "context": cf["retrieved_context"],
                        "answers": cf["predicted_answer"],
                        # "similarity": cf["similarity"]
                    },
                )


if __name__ == '__main__':
    data_path = BASE_PATH + "src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_filtered_final.jsonl"
    save_path = BASE_PATH + "src/data/squad/t5_squad_counterfactuals/rag_counterfactuals_complete_noise_min_filtered_final_2.jsonl"
    dataset = CounterfactualDataset(data_path, save_path)
    counterfactuals = dataset.get_counterfactuals()

    # print(counterfactuals)
    # set 1: according to paper [nf + mf]
    # set 2: only minimal without checking pred == actual
    # if very less in set1 , them nf + mf w/o answer check in min.
