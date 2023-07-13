"""
Utilities for data handling.
"""
import json
import logging
import os
import pandas as pd
import shutil
import re
import random
from tqdm import tqdm

from typing import Dict

logger = logging.getLogger(__name__)


def read_data(file_path: str,
              task_name: str,
              guid_as_int: bool = False):
  """
  Reads task-specific datasets from corresponding GLUE-style TSV files.
  """
  logger.warning("Data reading only works when data is in TSV format, "
                 " and last column as classification label.")

  # `guid_index`: should be 2 for SNLI, 0 for MNLI and None for any random tsv file.
  if task_name == "MNLI":
    return read_glue_tsv(file_path,
                        guid_index=0,
                        guid_as_int=guid_as_int)
  elif task_name == "SNLI":
    return read_glue_tsv(file_path,
                        guid_index=2,
                        guid_as_int=guid_as_int)
  elif task_name == "WINOGRANDE":
    return read_glue_tsv(file_path,
                        guid_index=0)
  elif task_name == "QNLI":
    return read_glue_tsv(file_path,
                        guid_index=0)
  else:
    raise NotImplementedError(f"Reader for {task_name} not implemented.")


def convert_tsv_entries_to_dataframe(tsv_dict: Dict, header: str) -> pd.DataFrame:
  """
  Converts entries from TSV file to Pandas DataFrame for faster processing.
  """
  header_fields = header.strip().split("\t")
  data = {header: [] for header in header_fields}

  for line in tsv_dict.values():
    fields = line.strip().split("\t")
    assert len(header_fields) == len(fields)
    for field, header in zip(fields, header_fields):
      data[header].append(field)

  df = pd.DataFrame(data, columns=header_fields)
  return df


def copy_dev_test(task_name: str,
                  from_dir: os.path,
                  to_dir: os.path,
                  extension: str = ".tsv"):
  """
  Copies development and test sets (for data selection experiments) from `from_dir` to `to_dir`.
  """
  if task_name == "MNLI":
    dev_filename = "dev_matched.tsv"
    test_filename = "dev_mismatched.tsv"
  elif task_name in ["SNLI", "QNLI", "WINOGRANDE"]:
    dev_filename = f"dev{extension}"
    test_filename = f"test{extension}"
  else:
    raise NotImplementedError(f"Logic for {task_name} not implemented.")

  dev_path = os.path.join(from_dir, dev_filename)
  if os.path.exists(dev_path):
    shutil.copyfile(dev_path, os.path.join(to_dir, dev_filename))
  else:
    raise ValueError(f"No file found at {dev_path}")

  test_path = os.path.join(from_dir, test_filename)
  if os.path.exists(test_path):
    shutil.copyfile(test_path, os.path.join(to_dir, test_filename))
  else:
    raise ValueError(f"No file found at {test_path}")


def read_jsonl(file_path: str, key: str = "pairID"):
  """
  Reads JSONL file to recover mapping between one particular key field
  in the line and the result of the line as a JSON dict.
  If no key is provided, return a list of JSON dicts.
  """
  df = pd.read_json(file_path, lines=True)
  records = df.to_dict('records')
  logger.info(f"Read {len(records)} JSON records from {file_path}.")

  if key:
    assert key in df.columns
    return {record[key]: record for record in records}
  return



def read_glue_tsv(file_path: str,
                  guid_index: int,
                  label_index: int = -1,
                  guid_as_int: bool = False):
  """
  Reads TSV files for GLUE-style text classification tasks.
  Returns:
    - a mapping between the example ID and the entire line as a string.
    - the header of the TSV file.
  """
  tsv_dict = {}

  i = -1
  with open(file_path, 'r') as tsv_file:
    for line in tqdm.tqdm([line for line in tsv_file]):
      i += 1
      if i == 0:
        header = line.strip()
        field_names = line.strip().split("\t")
        continue

      fields = line.strip().split("\t")
      label = fields[label_index]
      if len(fields) > len(field_names):
        # SNLI / MNLI fields sometimes contain multiple annotator labels.
        # Ignore all except the gold label.
        reformatted_fields = fields[:len(field_names)-1] + [label]
        assert len(reformatted_fields) == len(field_names)
        reformatted_line = "\t".join(reformatted_fields)
      else:
        reformatted_line = line.strip()

      if label == "-" or label == "":
        logger.info(f"Skippping line: {line}")
        continue

      if guid_index is None:
        guid = i
      else:
        guid = fields[guid_index] # PairID.
      if guid in tsv_dict:
        logger.info(f"Found clash in IDs ... skipping example {guid}.")
        continue
      tsv_dict[guid] = reformatted_line.strip()

  logger.info(f"Read {len(tsv_dict)} valid examples, with unique IDS, out of {i} from {file_path}")
  if guid_as_int:
    tsv_numeric = {int(convert_string_to_unique_number(k)): v for k, v in tsv_dict.items()}
    return tsv_numeric, header
  return tsv_dict, header


def convert_string_to_unique_number(string: str) -> int:
  """
  Hack to convert SNLI ID into a unique integer ID, for tensorizing.
  """
  id_map = {'e': '0', 'c': '1', 'n': '2'}

  # SNLI-specific hacks.
  if string.startswith('vg_len'):
    code = '555'
  elif string.startswith('vg_verb'):
    code = '444'
  else:
    code = '000'

  try:
    number = int(code + re.sub(r"\D", "", string) + id_map.get(string[-1], '3'))
  except:
    number = random.randint(10000, 99999)
    logger.info(f"Cannot find ID for {string}, using random number {number}.")
  return number