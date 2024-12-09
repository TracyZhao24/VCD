## Usage
### Environment Setup
Environment for Running LLaVA-NeXT
```bash
conda create -yn vcd python=3.9
conda activate vcd
cd VCD
pip install -r requirements.txt
```

Environment for Running LlaMA3.2-3B Instruct for Ensembling
```bash
conda create -yn amber python=3.9
conda activate amber
cd VCD/amber
pip install -r requirements.txt
```

For AMBER evaluation, we need to also download the en_core_web_lg model
```
python -m spacy download en_core_web_lg
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
We included the [AMBER repo](https://github.com/junyangwang0410/AMBER) as a subdirectory. The images can be found here: [LINK](https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view?usp=sharing)

Download and place the images inside a folder called `image` inside the `amber/data` folder


## Reproduce our Results
Since all of our methods are training free, we do not have any training scripts.
Our inference scripts can be found in the `experiments/cd_scripts folder`: `llava1.6_amber.bash`, `llava1.6_pope.bash`

### Running the POPE script
`answers-file` argument is the name of the file that you want your inference results to be written to.  
`type` argument determines which subset of the COCO dataset to use. The 3 options are "random", "popular", "adversarial".  
`use_cd` argument is a flag that defaults to true. Set this to false if you wish to run the baseline model without using any form of visual contrastive decoding.  
On line 23 of the bash script, specify the path to the evaluation file you would like to use based on the method you are experimenting with.

### Running the AMBER script
`answers-file` argument is the name of the file that you want your inference results to be written to.  
`use_cd` argument is a flag that defaults to true. Set this to false if you wish to run the baseline model without using any form of visual contrastive decoding.  
On line 24 of the bash script, the evaluation file is already specified to the amber python script.


### Evaluation Files 
Our evaluation files can be found in the `experiments/eval` folder. Use vcd env for all of these. 
1. `object_hallucination_vqa_llava.py`: use for baseline model, plain VCD, VCD with image segmentation experiments
2. `object_hallucination_vqa_llava_colorjitter.py`: use for color jitter experiments
3. `object_hallucination_vqa_llava_object_boxblur.py`: use for gaussian blur experiments
4. `object_hallucination_vqa_llava_amber.py`: use for AMBER benchmark for baseline model, plain VCD, VCD with image segmentation, color jitter, and gaussian blur

### Ensembling
We have two types of ensembling. We use `experiments/output/ensemble5.py` to ensemble the model results for POPE and `amber/llama_ensemble.py` to ensemble the generative results from the models when evaluated on AMBER.
1. `amber/llama_ensemble.py`: (Run with amber env) Specify the output files from models ran on AMBER that you want to ensemble. Default is vcd, blur 15, blur 35, color jitter, and image segment. You can configure by passing arguments `res1_file`, `res2_file`, `res3_file`, `res4_file`, and `res5_file` 
2. `experiments/output/ensemble5.py`: (Run with vcd env) Specify the output files from models ran on POPE that you want to ensemble. Default is vcd, blur 15, blur 35, color jitter, and image segment. You can configure by changing the variables in the file `file1`, `file2`, `file3`, `file4`, and `file5`



### Calculate Benchmark POPE Scores
After inference is complete, compute the Precision, Recall, F1, and Accuracy using the `eval_pope.py` script in the `experiments/eval` folder. Execute the script with the appropriate values for `gt_file`, `gen_file`, and `output_file` arguments and in the vcd env.

### Calculate Benchmark AMBER Scores
After inference is complete, compute the CHAIR, Cover, Hallucination, and Cog scores using the `inference.py` script in the `amber/` folder. In the amber env, execute the script where INPUT_PATH is the path to the file containing the generated outputs of a model ran on the AMBER dataset:
```bash
python inference.py --inference_data INPUT_PATH --evaluation_type g
```
