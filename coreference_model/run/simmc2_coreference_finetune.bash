# Example run:
# bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "object_indexes_in_descriptions='number'" 20 train ""

# The name of this experiment.
#name=$2
batch_size=$2
#read -a features <<< $3
features=$3
num_runs=1
epochs=$4
mode=$5
image_feature_file="all_gt_boxes"


# build input features and a suffix for the folder
#output_dir_features=""
#input_features="["
#for feature in "${features[@]}"; do
#    if [ "$output_dir_features" != "" ]; then
#        output_dir_features+='_'
#        input_features+=', '
#    fi
#    output_dir_features+="${feature}"
#    input_features+="'${feature}'"
#done
#input_features+="]"
#echo "${input_features}"


# Save logs and models under snap/simmc2; make backup.
output="snap/simmc2" # ${name}"
#mkdir -p $output/src
#cp -r src/* $output/src/
#cp $0 $output/run.bash


#PYTHONBREAKPOINT="pudb.set_trace"

#WANDB_MODE=disabled
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/simmc2_coreference_pl.py \
    --train train --valid dev --test devtest --num_runs $num_runs \
    --llayers 5 --xlayers 3 --rlayers 3 --numWorkers 8 \
    --loadLXMERT snap/pretrained/model \
    --batchSize "${batch_size}" --optim bert --lr 5e-5 --epochs $epochs \
	--simmc2_input_features "${features}" \
	--image_feature_file ${image_feature_file} \
    --tqdm --output $output --mode ${mode} ${@:6}

echo ''
echo 'Done!'
echo ''


# bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "img_bounding_boxes=False" 10 train "--final_layer sequential" && bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "previous_object_turns='all'" 10 train "--final_layer linear" && bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "object_indexes_in_descriptions='token_global'" 10 train "--final_layer linear" && bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "object_indexes_in_descriptions='token_global', previous_object_turns='all'" 10 train "--final_layer linear"


# Example experiments:

# base model without vision, just token ids: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "img_features=False, img_bounding_boxes=False, object_counts=False, object_indexes_in_descriptions='token_global', descriptions=False, previous_object_turns='all'" 10 train "--final_layer linear --ablation visual_encoder"

# ablation for lxrt encoder: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "" 10 train "--final_layer sequential --ablation lxrt_encoder"

# no bounding boxes: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "img_bounding_boxes=False" 10 test "--final_layer sequential --wandb_id 1irsbp8m --load coref-1irsbp8m-val@f1=0.582-step=2849-train@loss0.079-user_utterance=T-img_features=T-img_bounding_boxes=F-dialogue_history_turns=1-previous_object_turns=1-descriptions=T-object_indexes_in_descriptions=number-object_counts=T.ckpt"

# full with all objects: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "previous_object_turns='all'" 10 test "--final_layer linear --wandb_id 14iugqed --load coref-14iugqed-val@f1=0.614-step=2849-train@loss0.077-user_utterance=T-img_features=T-img_bounding_boxes=gold-dialogue_history_turns=1-previous_object_turns=all-descriptions=T-object_indexes_in_descriptions=number-object_counts=T.ckpt"

# global default: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "object_indexes_in_descriptions='token_global'" 10 test "--final_layer linear --wandb_id 2636fnjo --load coref-2636fnjo-val@f1=0.659-step=2849-train@loss0.067-user_utterance=T-img_features=T-img_bounding_boxes=gold-dialogue_history_turns=1-previous_object_turns=1-descriptions=T-object_indexes_in_descriptions=token_global-object_counts=T.ckpt"

# global with all: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "object_indexes_in_descriptions='token_global', previous_object_turns='all'" 10 train "--final_layer linear"

# no object counts: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "object_counts=False" 10 train "--final_layer sequential"

# global without visual feats: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "object_indexes_in_descriptions='token_global', img_features=False" 10 train "--final_layer linear"

# ablation with single objects: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "" 10 train "--final_layer sequential --ablation multiple_similar_objects"

# no bounding boxes: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "img_bounding_boxes=False" 10 train "--final_layer sequential"

# ablation without visual encoder: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "" 10 train "--final_layer sequential --ablation visual_encoder"

# refined with oracle descriptions: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "descriptions='gold'" 10 train "--final_layer sequential"

# global without descriptions: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "object_indexes_in_descriptions='token_global', descriptions=False" 10 train "--final_layer linear"

# no user utterance: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "user_utterance=False" 10 test "--final_layer sequential --wandb_id 1ex498zj --load coref-1ex498zj-val@f1=0.324-step=1994-train@loss0.174-user_utterance=F-img_features=T-img_bounding_boxes=gold-dialogue_history_turns=1-previous_object_turns=1-descriptions=T-object_indexes_in_descriptions=number-object_counts=T.ckpt"

# no prev object: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "previous_object_turns=False" 10 test "--final_layer sequential --wandb_id dxmdigt4 --load coref-dxmdigt4-val@f1=0.462-step=2849-train@loss0.097-user_utterance=T-img_features=T-img_bounding_boxes=gold-dialogue_history_turns=1-previous_object_turns=0-descriptions=T-object_indexes_in_descriptions=number-object_counts=T.ckpt"

# no descriptions: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "descriptions=False" 10 test "--final_layer sequential --wandb_id 3bz40f28 --load coref-3bz40f28-val@f1=0.545-step=2849-train@loss0.083-user_utterance=T-img_features=T-img_bounding_boxes=gold-dialogue_history_turns=1-previous_object_turns=1-descriptions=F-object_indexes_in_descriptions=number-object_counts=T.ckpt"

# no object IDs in descriptions: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "object_indexes_in_descriptions=False" 10 test "--final_layer sequential --wandb_id 39mprsaw --load coref-39mprsaw-val@f1=0.460-step=2849-train@loss0.097-user_utterance=T-img_features=T-img_bounding_boxes=gold-dialogue_history_turns=1-previous_object_turns=1-descriptions=T-object_indexes_in_descriptions=F-object_counts=T.ckpt"

# no prev system turn: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "dialogue_history_turns=False" 10 test "--final_layer sequential --wandb_id jpjy0mhj --load coref-jpjy0mhj-val@f1=0.581-step=2849-train@loss0.079-user_utterance=T-img_features=T-img_bounding_boxes=gold-dialogue_history_turns=0-previous_object_turns=1-descriptions=T-object_indexes_in_descriptions=number-object_counts=T.ckpt"

# no colours in descriptions: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "descriptions='asset_only'" 10 test "--final_layer sequential --wandb_id 2p8mil4x --load coref-2p8mil4x-val@f1=0.547-step=2564-train@loss0.091-user_utterance=T-img_features=T-img_bounding_boxes=gold-dialogue_history_turns=1-previous_object_turns=1-descriptions=asset_only-object_indexes_in_descriptions=number-object_counts=T.ckpt"

# no types in descriptions: bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "descriptions='colours_only'" 10 test "--final_layer sequential --wandb_id 14e8h5xn --load coref-14e8h5xn-val@f1=0.575-step=2849-train@loss0.080-user_utterance=T-img_features=T-img_bounding_boxes=gold-dialogue_history_turns=1-previous_object_turns=1-descriptions=colours_only-object_indexes_in_descriptions=number-object_counts=T.ckpt"
