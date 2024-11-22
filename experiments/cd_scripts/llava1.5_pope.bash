SCRIPT_DIR=$(dirname "$0")
seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"random"}
# model_path=${4:-"./checkpoints/llava-v1.5-7b"}
# model_path=${4:-"liuhaotian/llava-v1.5-7b"}
# model_path=${4:-"llava-hf/llava-v1.6-mistral-7b-hf"}
model_path=${4:-"liuhaotian/llava-v1.6-mistral-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.1}
noise_step=${7:-500}
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder="$SCRIPT_DIR/../data/coco/val2014"
else
  image_folder="$SCRIPT_DIR../data/gqa/images"
fi

python "$SCRIPT_DIR/../eval/object_hallucination_vqa_llava.py" \
--model-path ${model_path} \
--question-file "$SCRIPT_DIR/../data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json" \
--image-folder ${image_folder} \
--answers-file "$SCRIPT_DIR/../output/llava15_${dataset_name}_pope_${type}_answers_no_cd_seed${seed}.jsonl" \
--use_cd \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed}


