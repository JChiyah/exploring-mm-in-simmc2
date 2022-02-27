# Author: JChiyah 2021
# Date: 2021
#
# Example run:
# ./evaluation_simmc2_pipeline.sh 1 eval "user_utterance" 0
#
# Modify line 49 about load_path to load a fine-tuned model if downloading from GitHub repository

gpu=$1
model="todbert"
model_name="TODBERT/TOD-BERT-JNT-V1"
#output_dir=$4
mode=$2
batch_size=64
read -a features <<< $3
max_turns=$4
clean_previous=$5
train_data_ratio=$6


#declare -a features=('user_utterance' 'dialogue_history')

epoch=10
#batch_size=32

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

#output_dir="${model_name}/batch${batch_size}-turns${max_turns}"
output_dir="batch${batch_size}/turns${max_turns}-${output_dir_features}"
#full_output_dir="save/SIMMC2_MM_DISAMBIGUATION_${output_dir_features}/${output_dir}"
full_output_dir="save/SIMMC2_MM_DISAMBIGUATION/${model_name//[\/]/_}/${output_dir}"


if [ "$mode" == "train" ]; then
	mode="--do_train"
elif [ "$mode" == "eval" ]; then
    mode="--load_path ${full_output_dir}/run0/pytorch_model.bin"
else
	echo "error: choose either train or eval"
	exit 1
fi

if [ -z "$train_data_ratio" ]; then
	train_data_ratio=1
fi

# eval_by_step = num of training samples / batch_size, so that it evaluates at every epoch
eval_by_step=$(echo "(8512 / $batch_size) / (4 / $train_data_ratio)" | bc)
# at least evaluate 4 times per epoch

if [ "$clean_previous" == "clean" ]; then
	echo "Cleaning previous saved model at '${full_output_dir}'"
	rm -r "${full_output_dir}"
fi

CUDA_VISIBLE_DEVICES=$gpu python main.py \
	--my_model=multi_class_classifier \
	--data_path="../simmc2_data_generated/" \
	--dataset='["simmc2_mm_disambiguation"]' \
	--task_name="intent" \
	--simmc2_input_features="${input_features}" \
	--simmc2_max_turns="${max_turns}" \
	--simmc2_load_tmp_results \
	--earlystop="acc" \
	--output_dir="${full_output_dir}" \
	--task=nlu \
	--example_type=turn \
	--model_type=${model} \
	--model_name_or_path=${model_name} \
	--batch_size=${batch_size} \
	--usr_token=[USR] --sys_token=[SYS] \
	--epoch=$epoch --eval_by_step=$eval_by_step --warmup_steps=250 \
	--nb_runs=10 \
	--nb_evals=1 \
	--train_data_ratio="${train_data_ratio}" \
	${mode}
