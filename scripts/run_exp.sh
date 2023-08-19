BASE_PATH="/storage/ukp/work/sachdeva/research_projects/exp_calibration/src"


# List of model names
models=(
#"roberta-squad" \
#"roberta-squad-t5-squad-cfs-seed-42"  \
"roberta-squad-llama-context-rel-seed-42"  \
"roberta-squad-gpt-neox-context-rel-seed-42"  \
"roberta-squad-flan-ul2-context-rel-noise-seed-42"  \
)

datasets=(
#    "squad_adversarial"  \
#    "trivia_qa"  \
#    "hotpot_qa"  \
    "news_qa"  \
#    "natural_questions"  \
#    "bioasq"  \
)

exp_methods=(
"shap" \
"sc_attn"  \
"ig"  \
"attn"  \
"simple_grads"  \
)

TRAIN_SIZE=500

for MODEL_NAME in "${models[@]}"
do
    for DATASET in "${datasets[@]}"
    do
        echo "Calibrating model: ${MODEL_NAME}, dataset: ${DATASET}"
        echo "METHOD, ACC, AUC, MCE"
#        CONF_RES=$(python ${BASE_PATH}/calibration/baseline/modelling.py  \
#                --model_name ${MODEL_NAME}  \
#                --method "attn"   \
#                --train_size ${TRAIN_SIZE}   \
#                --dataset ${DATASET}  \
#                --do_maxprob 2>/dev/null | tail -1)
#            echo 'Conf,' ${CONF_RES}

        for EXP_METHOD in "${exp_methods[@]}"
        do
            EXP_RES=$(python ${BASE_PATH}/calibration/baseline/modelling.py  \
                --model_name ${MODEL_NAME}  \
                --method ${EXP_METHOD}   \
                --train_size ${TRAIN_SIZE}   \
                --dataset ${DATASET}  \
                --arg_n_tree 500  \
                --dense_features  \
                --arg_max_depth 20 2>/dev/null | tail -1)
            echo ${EXP_METHOD}',' ${EXP_RES}
        done
        echo "Finished calibrating for model: ${MODEL_NAME}, dataset: ${DATASET}"
    done
done
