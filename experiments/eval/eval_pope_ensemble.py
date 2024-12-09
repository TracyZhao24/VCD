import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--gt_file", type=str, default="../data/POPE/coco/coco_pope_popular.json")
parser.add_argument("--gen_files_folder", type=str, default="../output/pope/popular/")
parser.add_argument("--output_file", type=str, default="../output/pope/popular/pope_eval_combined_results.txt")
args = parser.parse_args()

# Specify the 10 generation files explicitly
gen_files_list = [
    "coco_ensemble_123.json",
    "coco_ensemble_124.json",
    "coco_ensemble_125.json",
    "coco_ensemble_134.json",
    "coco_ensemble_135.json",
    "coco_ensemble_145.json",
    "coco_ensemble_234.json",
    "coco_ensemble_235.json",
    "coco_ensemble_245.json",
    "coco_ensemble_345.json",
]

# Create full paths for the generation files
gen_files = [os.path.join(args.gen_files_folder, file) for file in gen_files_list]

# Open ground truth file
gt_file = [json.loads(q) for q in open(os.path.expanduser(args.gt_file), "r")]

# Prepare a container for combined results
combined_results = []

for gen_file_path in tqdm(gen_files, desc="Processing generation files"):
    # Open generated answers
    gen_file = [json.loads(q) for q in open(os.path.expanduser(gen_file_path), "r")]

    # Initialize metrics
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    unknown = 0
    total_questions = len(gt_file)
    yes_answers = 0

    num_examples = len(gen_file)

    # Compare answers
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

    # Calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    accuracy = (true_pos + true_neg) / total_questions if total_questions > 0 else 0
    yes_proportion = yes_answers / total_questions if total_questions > 0 else 0
    unknown_prop = unknown / total_questions if total_questions > 0 else 0

    # Append results to combined results
    combined_results.append({
        "file": os.path.basename(gen_file_path),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "yes_proportion": yes_proportion,
        "unknown_proportion": unknown_prop,
    })

# Write combined results to the output file
with open(args.output_file, "w") as output_file:
    for result in combined_results:
        output_file.write(f"Results for {result['file']}:\n")
        output_file.write(f"Precision: {result['precision']}\n")
        output_file.write(f"Recall: {result['recall']}\n")
        output_file.write(f"F1: {result['f1']}\n")
        output_file.write(f"Accuracy: {result['accuracy']}\n")
        output_file.write(f"Yes Proportion: {result['yes_proportion']}\n")
        output_file.write(f"Unknown Proportion: {result['unknown_proportion']}\n")
        output_file.write("\n")

print("All results combined into:", args.output_file)