import os

from transformers import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from transformers import AutoConfig, AutoTokenizer

from dataclasses import dataclass

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
# BASE_PATH="/home/sachdeva/projects/ukp/exp_calibration"
BASE_PATH="/storage/ukp/work/sachdeva/research_projects/exp_calibration"

# roberta-squad-llama-13b-v2-temp-0.7-seed-1
# roberta-squad-flan-ul2-v1-temp-0.7
# roberta-squad-gpt-jt-v2-temp-0.7-seed-0
# roberta-squad-gpt-neox-context-rel-seed-42
# roberta-squad-flan-ul2-context-rel-noise-seed-42

@dataclass
class InterpConfig:
    model_type: str = os.getenv("MODEL_TYPE", "roberta")
    model_name_or_path: str = f"{BASE_PATH}/{os.getenv('MODEL_NAME_OR_PATH', 'roberta-squad-gpt-neox-context-rel-seed-42')}"
    tokenizer_name_or_path: str = "roberta-base"
    dataset: str = os.getenv("DATASET", "trivia")
    dataset_config: str = "AddSent"
    output_dir: str = BASE_PATH + "/src/data/squad"
    train_file: str = None
    predict_file: str = BASE_PATH + "/src/data/"
    eval_batch_size: int = 1
    config_name: str = None
    max_seq_length: int = 384
    max_query_len: int = 64
    max_answer_len: int = 30
    do_lower_case: bool = True
    n_best_size: int = 20
    cache_dir: str = None
    seed: int = 42
    local_rank: int = -1
    threads: int = -1
    first_n_samples: int = 13000
    do_vis: bool = False
    no_cuda: bool = False
    n_gpu: int = 1
    percent: float = 0.3
    percent_augment: float = 0.15
    # interp_dir: str = f"{os.getenv('INTERP_DIR', 'exp_llama2_flan_ul2')}/shap/trivia/dev/roberta"
    # visual_dir: str = f"{os.getenv('INTERP_DIR', 'exp_llama2_flan_ul2')}/shap/trivia/dev/visual"
    interp_dir: str = "exp_roberta_gpt_neox_context_rel/shap/bioasq/dev/roberta"
    visual_dir: str = "exp_roberta_gpt_neox_context_rel/shap/bioasq/dev/visual"


def load_config_and_tokenizer(args):
    if args.dataset in ['simple', 'synth', 'comp']:
        tokenizer = SimBertTokenizer()
        if args.dataset == 'simple':
            config = SimBertConfig()
        elif args.dataset == 'comp':
            config = CompBertConfig()
        else:
            config = SynBertConfig()
    elif args.dataset in ['hpqa', 'squad', 'bioasq', 'newsqa', 'natq', 'trivia', 'hotpot', 'squad_adversarial']:
        config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:
        raise RuntimeError('Dataset not supported')

    return config, tokenizer
