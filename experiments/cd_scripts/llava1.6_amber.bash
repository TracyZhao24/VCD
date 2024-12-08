SCRIPT_DIR=$(dirname "$0")
seed=${1:-55}
dataset_name=${2:-"amber"}
type=${3:-"generative"}
model_path=${4:-"liuhaotian/llava-v1.6-mistral-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.1}
noise_step=${7:-500}
image_folder="/home/project/AMBER/data/image"

python "$SCRIPT_DIR/../eval/object_hallucination_vqa_llava_amber.py" \
--model-path ${model_path} \
--question-file "/home/project/AMBER/data/query/query_generative.json" \
--image-folder ${image_folder} \
--answers-file "$SCRIPT_DIR/../output/amber/${dataset_name}_amber_${type}_colorjitter_v1_no_vcd_${seed}.jsonl" \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed}


