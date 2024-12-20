import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--gt_file", type=str, default="../data/POPE/coco/coco_pope_random.json")
parser.add_argument("--gen_file", type=str, default="../output/pope/random/coco_ensemble5_random.json")
parser.add_argument("--output_file", type=str, default="../output/pope/random/coco_ensemble5_random.txt")
args = parser.parse_args()


# open ground truth answers
gt_file = [json.loads(q) for q in open(os.path.expanduser(args.gt_file), "r")]

# open generated answers
gen_file = [json.loads(q) for q in open(os.path.expanduser(args.gen_file), "r")]

# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
unknown = 0
total_questions = len(gt_file)
yes_answers = 0

num_examples = len(gen_file)

# compare answers
for index, line in enumerate(gt_file[:num_examples]):
    idx = line["question_id"]
    gt_answer = line["label"]
    assert idx == gen_file[index]["question_id"]
    gen_answer = gen_file[index]["text"]
    # convert to lowercase
    gt_answer = gt_answer.lower()
    gen_answer = gen_answer.lower()
    # strip
    gt_answer = gt_answer.strip()
    gen_answer = gen_answer.strip()
    # pos = 'yes', neg = 'no'
    if gt_answer == 'yes':
        if 'yes' in gen_answer:
            true_pos += 1
            yes_answers += 1
        else:
            false_neg += 1
    elif gt_answer == 'no':
        if 'no' in gen_answer:
            true_neg += 1
        else:
            yes_answers += 1
            false_pos += 1
    else:
        print(f'Warning: unknown gt_answer: {gt_answer}')
        unknown += 1
# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f1 = 2 * precision * recall / (precision + recall)
accuracy = (true_pos + true_neg) / total_questions
yes_proportion = yes_answers / total_questions
unknown_prop = unknown / total_questions

# report results and write to file
with open(args.output_file, "w") as file:
    file.write(f"results for {args.gen_file}\n")
    file.write(f"Precision: {precision}\n")
    file.write(f"Recall: {recall}\n")
    file.write(f"F1: {f1}\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"yes: {yes_proportion}\n")
    file.write(f"unknown: {unknown_prop}\n")
    
print(f"Reporting results for {args.gen_file}")
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {f1}')
print(f'Accuracy: {accuracy}')
print(f'yes: {yes_proportion}')
print(f'unknown: {unknown_prop}')