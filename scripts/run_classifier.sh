RUN_SQUAD_EXP () {
    DATASET="squad"
    TRAIN_SIZE=${1:-500}
    PATH_SPEC="./src/results"
    echo "${DATASET}-${TRAIN_SIZE}"
    echo "METHOD, AUC, ACC, F1@25, F1@50, F1@75"
    CONF_RES=$(python ./src/modelling.py --train_size ${TRAIN_SIZE} --dataset ${DATASET} --save_path ${PATH_SPEC} --path ${PATH_SPEC} --do_maxprob 2>/dev/null | tail -1)
    echo 'Conf,'${CONF_RES}
    KAM_RES=$(python ./src/modelling.py --train_size ${TRAIN_SIZE} --dataset ${DATASET} --save_path ${PATH_SPEC} --path ${PATH_SPEC} --do_baseline --arg_n_tree 300 --arg_max_depth 6  2>/dev/null  | tail -1)
    echo 'KAMATH,'${KAM_RES}
    BOW_RES=$(python ./src/modelling.py --train_size ${TRAIN_SIZE} --dataset ${DATASET} --save_path ${PATH_SPEC} --path ${PATH_SPEC} --do_bow --arg_n_tree 200 --arg_max_depth 20 2>/dev/null  | tail -1)
    echo 'BOW,'${BOW_RES}
    EXPL_RES=$(python ./src/modelling.py --train_size ${TRAIN_SIZE} --dataset ${DATASET} --save_path ${PATH_SPEC} --path ${PATH_SPEC} --arg_n_tree 300 --arg_max_depth 20 2>/dev/null | tail -1)
    echo 'ShapCal,'${EXPL_RES}
}

RUN_SQUAD_EXP 500

