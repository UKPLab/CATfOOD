import jsonlines
from tqdm import tqdm
import pickle


def merge_jsonl_files(files, out_file):
    with jsonlines.open(out_file, "w") as writer:
        for file in tqdm(files):
            with jsonlines.open(file) as reader:
                for obj in reader:
                    writer.write(obj)


def _load_binary_file(file):
    with open(file, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # base_path = "/storage/ukp/work/sachdeva/research_projects/exp_calibration"
    # file_base_path = base_path + "/src/data/squad/rag_predictions_with_answers_noise_filtered_20p_chunk"
    # chunks = 5
    # files = [file_base_path + str(i+1) + ".jsonl" for i in range(chunks)]
    # out_file = base_path + "/src/data/squad/squad_counterfactuals_noise_filtered.jsonl"
    # merge_jsonl_files(files, out_file)

    base_path = "/"
    file = base_path + "/src/data/squad_adv_new/calibration_data_final.bin"
    data = _load_binary_file(file)
    print(data)
