## Usage
### Environment Setup
```bash
conda create -yn vcd python=3.9
conda activate vcd
cd VCD
pip install -r requirements.txt
```

### How to Use VCD in LVLMs

The two core functions of VCD, adding noise to images and generating text based on VCD sampling, are found in the `vcd_utils` folder. 

To help you get started quickly, here's an example using LLaVA on how to replace the conventional sampling method with the VCD method during generation:
1. Add the following at the beginning of the start-up script:
```python
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()
```
The `evolve_vcd_sampling` function replaces the sampling function in the transformers library. The modified sampling function includes an option for visual contrastive decoding, while keeping the rest unchanged.

2. Slightly modify `llava_llama.py`:

   a. Add contrastive decoding parameters in the `LlavaLlamaForCausalLM` class's `forward` function to avoid exceptions in `model.generate`.
   
   b. Add the `prepare_inputs_for_generation_cd` function.

3. Add noise to the image:
```python
from vcd_utils.vcd_add_noise import add_diffusion_noise
image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
```
set the hyperparameter in the `generate` function:
```python
output_ids = model.generate(
    input_ids,
    images=image_tensor.unsqueeze(0).half().cuda(),
    images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
    cd_alpha = args.cd_alpha,
    cd_beta = args.cd_beta,
    do_sample=True)
```

## Obtain the Datasets
### POPE
We evaluated our methods using the POPE benchmark on the COCO 2014 Val dataset, which can be found here: http://images.cocodataset.org/zips/val2014.zip. 
To download and extract the files:
```
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d val2014
```

### AMBER
TODO

## Reproduce our Results
Since all of our methods are training free, we do not have any training scripts.
Our inference scripts can be found in the `experiments/cd_scripts folder`: `llava1.6_amber.bash`, `llava1.6_pope.bash`

### Running the POPE script
`answers-file` argument is the name of the file that you want your inference results to be written to.  
`type` argument determines which subset of the COCO dataset to use. The 3 options are "random", "popular", "adversarial".  
`use_cd` argument is a flag that defaults to true. Set this to false if you wish to run the baseline model without using any form of visual contrastive decoding.  
In line 23 of the bash script, specify the path to the evaluation file you would like to use based on the method you are experimenting with.

### Running the AMBER script
TODO

### Evaluation Files 
Our evaluation files can be found in the `experiments/eval/` folder.  
1. `object_hallucination_vqa_llava.py`: use for baseline model, plain VCD, VCD with image segmentation experiments
2. `object_hallucination_vqa_llava_colorjitter.py`: use for color jitter experiments
3. `object_hallucination_vqa_llava_object_boxblur.py`: use for gaussian blur experiments
4. `object_hallucination_vqa_llava_amber.py`: use for AMBER benchmark


### Calculate Benchmark Scores
After inference is complete, compute the Precision, Recall, F1, and Accuracy using the `eval_pope.py` script in the `experiments/eval` folder. Execute the script with the appropriate values for `gt_file`, `gen_file`, and `output_file` arguments. 
