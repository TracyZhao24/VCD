import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

# import kornia
# from transformers import set_seed
import transformers
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()


def eval_model(args):
    os.environ['HF_HOME'] = '/.cache/huggingface'
    os.environ["TRANSFORMERS_CACHE"] = '/.cache/huggingface'
    # make sure it is looking in the right cache
    transformers_cache = os.getenv("TRANSFORMERS_CACHE") 
    hf_home = os.getenv("HF_HOME")
    print(transformers_cache, hf_home)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # print(image_processor, tokenizer, model)
    # processor = transformers.LlavaNextProcessor.from_pretrained(model_name)
    # image_processor = processor.image_processor
    # tokenizer = processor.tokenizer
    # model = transformers.LlavaNextForConditionalGeneration.from_pretrained(
    #     model_name, 
    #     torch_dtype=torch.float16,
    #     low_cpu_mem_usage=True,
    #     load_in_4bit=True,
    #     # use_flash_attention_2=True
    # )
    # context_len = 2048
    print("model loaded")

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # batch writes to the file
    results = []
    buffer_size = 100
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if hasattr(model.config, 'mm_use_im_start_end') and model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        # ValueError: Unable to create tensor, you should probably activate padding with 'padding=True' to have batched tensors with the same length.
        # image_tensor = image_processor.preprocess(image, padding=True, return_tensors='pt')['pixel_values'][0]
        image_tensor = image_processor(image, return_tensors='pt')['pixel_values'][0]
        
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None      

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        # min_token_len = min(input_ids.shape[1], output_ids.shape[1])
        # print("input id shape: ", input_ids.shape)
        # print("output id shape: ", output_ids.shape)
        # print(output_ids[:, :input_token_len])

        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # n_diff_input_output = (input_ids[:, :min_token_len] != output_ids[:, :min_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        results.append({
            "question_id": idx,  # Replace `i` with `idx` if you have it defined
            "prompt": cur_prompt,  # Replace with your variable `cur_prompt`
            "text": outputs,       # Replace with your variable `outputs`
            "model_id": model_name,  # Replace with your variable `model_name`
            "image": image_file,    # Replace with your variable `image_file`
            "metadata": {}
        })

        # Write batch to file
        if len(results) >= buffer_size:
            # with open("output.jsonl", "a") as ans_file:  # Open in append mode
            ans_file.writelines(json.dumps(obj) + "\n" for obj in results)
            ans_file.flush()
            results = []  # Clear the batch

    # Write any remaining data
    if results:
        # with open("output.jsonl", "a") as ans_file:
        ans_file.writelines(json.dumps(obj) + "\n" for obj in results)
    ans_file.flush()
    ans_file.close()
        
        # ans_file.write(json.dumps({"question_id": idx,
        #                         "prompt": cur_prompt,
        #                         "text": outputs,
        #                         "model_id": model_name,
        #                         "image": image_file,
        #                         "metadata": {}}) + "\n")
    #     ans_file.flush()
    # ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    transformers.set_seed(args.seed)
    eval_model(args)
