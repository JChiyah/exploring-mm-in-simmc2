# Example run:
# bash run/simmc2_finetune.bash 0 lxr955 disambiguation 64 "user_utterance image"

# The name of this experiment.
name=$2
subtask=$3
batch_size=$4
read -a features <<< $5


# build input features and a suffix for the folder
output_dir_features=""
input_features="["
for feature in "${features[@]}"; do
    if [ "$output_dir_features" != "" ]; then
        output_dir_features+='_'
        input_features+=', '
    fi
    output_dir_features+="${feature}"
    input_features+="'${feature}'"
done
input_features+="]"
echo "${input_features}"


# Save logs and models under snap/simmc2; make backup.
output="snap/simmc2/${subtask}_${name}_${batch_size}_${output_dir_features}"
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash


# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/simmc2_${subtask}.py \
    --train train --valid dev --test devtest --num_runs 10 \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT snap/pretrained/model \
    --batchSize $batch_size --optim bert --lr 5e-5 --epochs 10 \
	--simmc2_input_features="${input_features}" \
    --tqdm --output $output ${@:6}

echo ''
echo 'Done!'
echo ''

#exit_status=$?
#
#if [ $exit_status -eq 0 ]; then
#	echo 'Testing model...'
#
#	CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
#    python src/tasks/simmc2_disambiguation.py \
#    --tiny --train train --valid dev --test devtest  \
#    --llayers 9 --xlayers 5 --rlayers 5 \
#    --batchSize 64 --optim bert --lr 5e-5 --epochs 4 \
#    --load $output/BEST \
#    --tqdm --output "${output}_test_results" ${@:3}
#
#fi
