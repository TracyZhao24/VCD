import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json
import argparse
import math

import sys
import os
sys.path.insert(0, (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

os.environ['HF_HOME'] = '/.cache/huggingface'
os.environ["TRANSFORMERS_CACHE"] = '/.cache/huggingface'

def load_model():    
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # config = AutoConfig.from_pretrained("meta-llma/Llama-3.2-3B", force_download=True)
    # print(config.to_dict())
    # config.rope_scaling = {"type": "linear", "factor": 32.0}
    # print(config.to_dict())

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=device,
        load_in_4bit=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


messages = [
    # {"role": "system", "content": "You are a chatbot who responds very shortly."},
    {"role": "user", "content": "When was UCLA founded?"},
]

def run_model(model, tokenizer, responses, num_responses=5, max_new_tokens=500, verbose=False):
    # TODO: Prepare the input text using the tokenizer's apply_chat_template (Do not tokenize the text yet)
    # sys_prompt = f'''You are helping me ensemble text output from {num_responses} different responses.
    #         Your output should be a similar length to one of the responses. It should be an ensemble of the {num_responses} responses.
    #         Only choose words or objects mentioned by a majority of the responses. 
    #         Only preserve elements and objects that are most commonly mentioned in a majority of responses.
    #         Do not include content only mentioned in one response.

    #         Do not generate any additional text. Only generate the ensembled response.

    #         '''
    sys_prompt = f'''You are helping me ensemble text output from {num_responses} different responses.
        It should be an ensemble of the {num_responses} responses.
        Only choose words or objects mentioned by a majority of the responses. 
        Only preserve elements and objects that are most commonly mentioned in a majority of responses.
        Do not include content mentioned by {math.floor(num_responses/2)} or less of the responses.
        Do not include content only mentioned in one response.
        Your output should be a similar length to one of the responses.

        Do not generate any additional text. Only generate the ensembled response.
        '''

    if num_responses == 3:
      prompt = sys_prompt + f'''
            Example 1:
            Response 1 -> A woman is sitting at a table with a cup of coffee.
            Response 2 -> A woman in a café drinking coffee and reading a book.
            Response 3 -> A person enjoying a peaceful moment in a café, sipping coffee while reading a book.
            Ensembled response -> A women is drinking coffee and reading a book.

            Example 2: 
            Response 1 -> Yes
            Response 2 -> Yes
            Response 3 -> No
            Ensembled response -> Yes

            Example 3:
            Response 1 -> Yes
            Response 2 -> No
            Response 3 -> No
            Ensembled response -> No

            Prompt:
            Response 1 -> {responses[0]}
            Response 2 -> {responses[1]}
            Response 3 -> {responses[2]}

            Ensembled response ->'''
    elif num_responses == 4:
        prompt = sys_prompt + f'''
            Example 1:
            Response 1 -> yes
            Response 2 -> yes
            Response 3 -> yes
            Response 4 -> yes
            Ensembled response -> yes

            Example 2:
            Response 1 -> yes
            Response 2 -> yes
            Response 3 -> yes
            Response 4 -> no
            Ensembled response -> yes

            Example 3:
            Response 1 -> no
            Response 2 -> no
            Response 3 -> yes
            Response 4 -> no
            Ensembled response -> no

            Example 4:
            Response 1 -> no
            Response 2 -> no
            Response 3 -> no
            Response 4 -> no
            Ensembled response -> no

            Example 5:
            Response 1 -> A woman is sitting at a table with a cup of coffee.
            Response 2 -> A woman in a café drinking coffee and reading a book.
            Response 3 -> A person enjoying a peaceful moment in a café, sipping coffee while reading a book.
            Response 4 -> A young woman drinks coffee as she reads the weekly newspaper and a crowded restaurant. The restaurant is decorated to celebrate Christmas.
            Ensembled response -> A woman is drinking coffee and reading a book.

            Example 6:
            Response 1 -> dog, cat, bird, lizard, fish
            Response 2 -> fish, cat, dog, pig, horse, bird
            Response 3 -> cat, sheep, dog, bird, wolf, table
            Response 4 -> fish, dog, cat, chair, chicken
            Ensembled response -> dog, cat, bird, fish

            Example 7:
            Response 1 -> This is an outdoor flower shop. There is a bouquet of balloon flowers amongst many bouquets of real flowers. The bouquets are on display on tiered shelves.
            Response 2 -> It's a beautiful afternoon as many customers admire beautiful flowers on display at an outdoor flower shop. A singular bouquet of balloon flowers in the middle draws the most attention.
            Response 3 -> An outdoor flower shop has bouquets of flowers on display on tiered shelves. One of the bouquets is a bouquet of balloon flowers while the rest contain real flowers.
            Response 4 -> On a bright morning, a flower shop has their best flowers and a flower balloon bouquet on display.
            Ensembled response -> An outdoor flower shop has bouquets of flowers and a singular flower balloon bouquet on display.

            Prompt:
            Response 1 -> {responses[0]}
            Response 2 -> {responses[1]}
            Response 3 -> {responses[2]}
            Response 4 -> {responses[3]}

            Ensembled response ->'''
    else:
        prompt = sys_prompt + f'''
            Example 1:
            Response 1 -> yes
            Response 2 -> yes
            Response 3 -> yes
            Response 4 -> yes
            Response 5 -> yes
            Ensembled response -> yes

            Example 2:
            Response 1 -> yes
            Response 2 -> yes
            Response 3 -> yes
            Response 4 -> yes
            Response 5 -> no
            Ensembled response -> yes

            Example 3:
            Response 1 -> yes
            Response 2 -> yes
            Response 3 -> yes
            Response 4 -> no
            Response 5 -> no
            Ensembled response -> yes

            Example 4:
            Response 1 -> yes
            Response 2 -> yes
            Response 3 -> no
            Response 4 -> yes
            Response 5 -> no
            Ensembled response -> yes

            Example 5:
            Response 1 -> no
            Response 2 -> no
            Response 3 -> yes
            Response 4 -> no
            Response 5 -> yes
            Ensembled response -> no

            Example 6:
            Response 1 -> no
            Response 2 -> no
            Response 3 -> yes
            Response 4 -> no
            Response 5 -> no
            Ensembled response -> no

            Example 7:
            Response 1 -> no
            Response 2 -> no
            Response 3 -> no
            Response 4 -> yes
            Response 5 -> yes
            Ensembled response -> no

            Example 8:
            Response 1 -> no
            Response 2 -> no
            Response 3 -> no
            Response 4 -> no
            Response 5 -> no
            Ensembled response -> no

            Example 9:
            Response 1 -> A woman is sitting at a table with a cup of coffee.
            Response 2 -> A woman in a café drinking coffee and reading a book.
            Response 3 -> A person enjoying a peaceful moment in a café, sipping coffee while reading a book.
            Response 4 -> A young woman drinks coffee as she reads the weekly newspaper and a crowded restaurant. The restaurant is decorated to celebrate Christmas.
            Response 5 -> On a cold winter day, a woman drinks coffee while reading a book. She sits on a cozy couch next to a grand fireplace. 
            Ensembled response -> A woman is drinking coffee and reading a book.

            Example 10:
            Response 1 -> dog, cat, bird, lizard, fish
            Response 2 -> sheep, dog, bird, lizard, crocodile
            Response 3 -> fish, cat, dog, pig, horse
            Response 4 -> cat, sheep, dog, bird, wolf, table
            Response 5 -> fish, dog, cat, chair, chicken
            Ensembled response -> dog, cat, bird, fish

            Example 11:
            Response 1 -> This is an outdoor flower shop. There is a bouquet of balloon flowers amongst many bouquets of real flowers. The bouquets are on display on tiered shelves.
            Response 2 -> It's a beautiful afternoon as many customers admire beautiful flowers on display at an outdoor flower shop. A singular bouquet of balloon flowers in the middle draws the most attention.
            Response 3 -> An outdoor flower shop has bouquets of flowers on display on tiered shelves. One of the bouquets is a bouquet of balloon flowers while the rest contain real flowers.
            Response 4 -> There are bouquets of flowers on display at an outdoor flower shop. A friendly worker jokes with customers as they view the bouquet of balloon flowers.
            Response 5 -> On a bright morning, a flower shop has their best flowers and a flower balloon bouquet on display.
            Ensembled response -> An outdoor flower shop has bouquets of flowers and a singular flower balloon bouquet on display.

            Prompt:
            Response 1 -> {responses[0]}
            Response 2 -> {responses[1]}
            Response 3 -> {responses[2]}
            Response 4 -> {responses[3]}
            Response 5 -> {responses[4]}

            Ensembled response ->'''

    # 5 responses
    input_text = tokenizer.apply_chat_template(
        [{
            "role": "user",
            "content": prompt
        }],
        add_generation_prompt=True,
        return_tensors="pt"
    )
    if verbose: print("\n###input_text:###\n", input_text)
    # TODO: Tokenize the input text and transfer it to the appropriate device
    input_ids = input_text.to(model.device)

    if verbose: print("\n###input_ids:###\n", input_ids)
    # TODO: Generate a response using the model. Ensure do_sample is False.
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # TODO: Decode the output and return the response without special tokens
    response = output[0][input_ids.shape[-1]:]
    assistant_response = tokenizer.decode(response, skip_special_tokens=True)
    if verbose: print("\n###response:###\n", response)
    # print("\n###############Assistant Response: ################# \n", assistant_response,"\n ################ END RESPONSE ####################")
    return assistant_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res1_file", type=str, default="amber_generative_vcd_15_box_blur_55.jsonl")
    parser.add_argument("--res2_file", type=str, default="amber_generative_vcd_15_box_blur_55.jsonl")
    parser.add_argument("--res3_file", type=str, default='amber_generative_with_vcd_55.jsonl')
    parser.add_argument("--res4_file", type=str, default='amber_generative_colorjitter_v1_no_vcd_55.jsonl')
    parser.add_argument("--res5_file", type=str, default='amber_generative_vcd_and_segment_55.jsonl')
    parser.add_argument("--ensemble_name", type=str, default="vcd_blur15_blur35_colorjitterv1_segment_prompt3")
    args = parser.parse_args()


    # Parse input files
    # answers_file_path = '../experiments/output/amber/vcd_blur15_blur35_segment_colorjitterv1_ensemble.jsonl'
    answers_file_path = '../experiments/output/amber/'+args.ensemble_name+'_ensemble.jsonl'


    os.makedirs(os.path.dirname(answers_file_path), exist_ok=True)
    answers_file = open(answers_file_path, 'w')

    model_res_dir = '../experiments/output/amber/'
    blur15_file = open(model_res_dir+args.res1_file)
    blur15_res = [json.loads(line) for line in blur15_file]

    blur35_file = open(model_res_dir+args.res2_file)
    blur35_res = [json.loads(line) for line in blur35_file]

    vcd_file = open(model_res_dir+args.res3_file)
    vcd_res = [json.loads(line) for line in vcd_file]

    colorjitter_file = open(model_res_dir+args.res4_file)
    colorjitter_res = [json.loads(line) for line in colorjitter_file]

    segment_file = open(model_res_dir+args.res5_file)
    segment_res = [json.loads(line) for line in segment_file]

    model_res = [vcd_res, blur15_res, blur15_res, colorjitter_res, segment_res]

    n = min(len(vcd_res), len(blur15_res), len(blur35_res), len(segment_res), len(colorjitter_res))

    model, tokenizer = load_model()

    results = []
    buffer_size = 10
    for i in range(n):
        responses = [model[i]['text'] for model in model_res]
        ensembled_response = run_model(model, tokenizer, responses, num_responses=5)
    
        results.append({
            "question_id": model_res[0][i]['question_id'],  
            "prompt": model_res[0][i]['prompt'],  
            "text": ensembled_response,       
            "model_id": "ensemble_"+args.ensemble_name,  
            "image": model_res[0][i]['image'],    
            "metadata": {}
        })

        # Write batch to file
        if len(results) >= buffer_size:
            answers_file.writelines(json.dumps(obj) + "\n" for obj in results)
            answers_file.flush()
            results = []  # Clear the batch
    if results:
        answers_file.writelines(json.dumps(obj) + "\n" for obj in results)
    answers_file.flush()
    answers_file.close()


if __name__ == "__main__":
    main()