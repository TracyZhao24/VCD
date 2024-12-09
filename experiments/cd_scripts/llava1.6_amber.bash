SCRIPT_DIR=$(dirname "$0")
answers_file=${1:-}
use_cd=${2:-"true"}  # Default to "true"
type=${3:-"generative"}
seed=${4:-55}
dataset_name=${5:-"amber"}
model_path=${6:-"liuhaotian/llava-v1.6-mistral-7b"}
cd_alpha=${7:-1}
cd_beta=${8:-0.1}
noise_step=${9:-500}
image_folder="$SCRIPT_DIR/../../amber/data/image"

if [ -z "$answers_file" ]; then
  echo "Error: answers_file is required as the first argument."
  exit 1
fi

if [[ $use_cd == "true" ]]; then
  use_cd_flag="--use_cd"
else
  use_cd_flag=""
fi

python "$SCRIPT_DIR/../eval/object_hallucination_vqa_llava_amber.py" \
--model-path ${model_path} \
--question-file "$SCRIPT_DIR/../../amber/data/query/query_generative.json" \
--image-folder ${image_folder} \
--answers-file "$SCRIPT_DIR/../output/amber/${dataset_name}_${type}_${answers_file}_${seed}.jsonl" \
${use_cd_flag} \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed}


