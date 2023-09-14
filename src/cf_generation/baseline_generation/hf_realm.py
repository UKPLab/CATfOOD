import os
import torch
import numpy as np
from typing import List
from transformers import (
    RealmRetriever,
    RealmTokenizer,
    RealmConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from src.rag.modelling_realm import RealmForOpenQA

device = "cuda" if torch.cuda.is_available() else "cpu"

retriever = RealmRetriever.from_pretrained("google/realm-orqa-nq-openqa")
tokenizer = RealmTokenizer.from_pretrained("google/realm-orqa-nq-openqa")
tokenizer.add_tokens("<hl>")
# tokenizer.add_special_tokens(
#         {"additional_special_tokens": ["<hl>"]}
#     )
hl_token_id = tokenizer.convert_tokens_to_ids(["<hl>"])[0]
config = RealmConfig.from_pretrained("google/realm-orqa-nq-openqa")
# print(config)
config.reader_beam_size = 5
model = RealmForOpenQA.from_pretrained(
    "google/realm-orqa-nq-openqa", retriever=retriever, config=config
)
# model.resize_token_embeddings(len(tokenizer))
model.to(device)

question = "Where did the super bowl 50 take place?"
question_ids = tokenizer([question], return_tensors="pt").to(device)
# answer_ids = tokenizer(
#     ["alan mathison turing"],
#     add_special_tokens=False,
#     return_token_type_ids=False,
#     return_attention_mask=False,
# ).input_ids

reader_output = model(**question_ids, return_dict=True)
# print(reader_output)
print(reader_output.input_ids)
print(hl_token_id)
start_positions = reader_output.reader_output.start_pos.cpu().numpy()
end_positions = reader_output.reader_output.end_pos.cpu().numpy()
# add hl_token_id at tensor index start position and end position
highlight_input_ids = reader_output.input_ids.cpu().numpy()
inputs: List = []
for idx, (start, end) in enumerate(zip(start_positions, end_positions)):
    start = int(start)
    end = int(end)
    # get sep idx
    sep_idx = np.where(highlight_input_ids[idx, :start] == tokenizer.sep_token_id)[0][
        -1
    ]
    inputs.append(
        np.concatenate(
            [
                highlight_input_ids[idx, sep_idx:start],
                [hl_token_id],
                highlight_input_ids[idx, start : end + 1],
                [hl_token_id],
                highlight_input_ids[idx, end + 1 :],
            ],
            axis=0,
        )
    )


predicted_answer = tokenizer.batch_decode(
    reader_output.predicted_answer_ids, skip_special_tokens=True
)
contexts = tokenizer.batch_decode(
    torch.tensor(np.array(inputs), device=device),
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)
# loss = reader_output.loss
print("==============================")
print(contexts)
print("==============================")
# print(loss)

retrieved_blocks = np.take(
    retriever.block_records,
    indices=reader_output.retrieved_block_ids.detach().cpu().numpy(),
    axis=0,
)
print(retrieved_blocks)

# print(retriever.block_records[0])
# print(retriever.block_records[101])

# text_pair = []
# for retrieved_block in retrieved_blocks:
#     text_pair.append(retriever.block_records[retrieved_block])
#
# for text in text_pair:
#     if "alan mathison turing" in text.decode("utf-8"):
#         print('====================')
#         print("found")
#         print(text)
#         break

# context = "The Super Bowl is currently played in early February (the game originally took place in early " \
#           "to mid-January), culminating a season that generally begins in September of the previous " \
#           "calendar year. For example, Super Bowl 50, which was played on February 7, 2016, determined " \
#           "the league champion for the 2015 NFL season. The years shown below refer to the season, not " \
#           "the date that the Super Bowl was actually played. The Pittsburgh Steelers (1974, 1975, 1978, " \
#           "1979, 2005, 2008) are first with six Vince Lombardi Trophies. The Dallas Cowboys (1971, 1977, " \
#           "1992, 1993, 1995), the San Francisco 49ers (1981, 1984, 1988, 1989, 1994), and the New " \
#           "England Patriots (2001, 2003, 2004, 2014, 2016) are tied for second with 5 each. The Green " \
#           "Bay Packers (1966, 1967, 1996, 2010) and the New York Giants (1986, 1990, 2007, 2011) are " \
#           "tied for fifth with four each. The Oakland Raiders (1976, 1980, 1983), the Washington " \
#           "Redskins (1982, 1987, 1991) and the Denver Broncos (1997, 1998, 2015) are tied for " \
#           "seventh with three each."

# x = torch.tensor([[8, 2, 3],
#                   [4, 5, 6]])
#
# print(torch.max(x, dim=0).values)
# print(torch.argmax(torch.max(x, dim=0).values))

# for answer, block in zip(predicted_answer, retrieved_blocks):
#     pass
#
BASE_PATH = os.getenv("PYTHONPATH", "/home/sachdeva/projects/exp_calibration/")
model_path = BASE_PATH + "t5-large-squad-qg-seed-42"
tokenizer = AutoTokenizer.from_pretrained("t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.to(device)
model.eval()
hl_token = "<hl>"
prefix = "generate question: "

prepared_inputs = [prefix + context for context in contexts]
features = tokenizer(
    prepared_inputs,
    max_length=640,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
).to(device)
# print(features)
outputs = model.generate(**features, max_length=128, num_beams=15, early_stopping=True)
dec_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("The predicted questions are: ", dec_preds)
print("The predicted answers are: ", predicted_answer)
