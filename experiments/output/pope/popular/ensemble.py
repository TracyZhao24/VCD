import json
from collections import Counter

def majority_vote_consolidation_plaintext(file1, file2, file3, output_file):
    """
    Consolidates three plaintext files (line-delimited JSON) based on majority voting for the 'text' field.

    Args:
    file1, file2, file3: Input plaintext files containing JSON entries.
    output_file: Path to the output plaintext file.
    """
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(file3, 'r') as f3:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        lines3 = f3.readlines()

    # Ensure all files have the same number of lines
    assert len(lines1) == len(lines2) == len(lines3), "Files must have the same number of entries."

    consolidated_data = []

    for line1, line2, line3 in zip(lines1, lines2, lines3):
        entry1 = json.loads(line1.strip())
        entry2 = json.loads(line2.strip())
        entry3 = json.loads(line3.strip())

        assert entry1["question_id"] == entry2["question_id"] == entry3["question_id"], \
            "Question IDs do not match between files."

        # Gather the 'text' votes
        votes = [entry1["text"], entry2["text"], entry3["text"]]

        # Determine the majority vote
        majority_vote = Counter(votes).most_common(1)[0][0]

        # Consolidate the entry
        consolidated_entry = {
            "question_id": entry1["question_id"],
            "prompt": entry1["prompt"],  # Assumes prompt is identical across files
            "text": majority_vote,
            "model_id": entry1["model_id"],  # Assumes model_id is identical across files
            "image": entry1["image"],  # Assumes image is identical across files
            "metadata": entry1["metadata"]  # Assumes metadata is identical across files
        }

        consolidated_data.append(consolidated_entry)

    # Write the consolidated data to the output file as line-delimited JSON
    with open(output_file, 'w') as outfile:
        for entry in consolidated_data:
            outfile.write(json.dumps(entry) + '\n')

# File paths
file1 = 'coco_pope_popular_answers_vcd_seed55.jsonl'
file2 = 'coco_pope_popular_answers_with_35_box_blur_cd_seed55.jsonl'
file4 = 'coco_pope_popular_answers_with_colorjitter_9k_seed55.jsonl'
file5 = 'coco_pope_popular_answers_with_segment_cd_seed55.jsonl'
file3 = 'coco_pope_popular_boxblur15_seed55.jsonl'
# output_file = 'coco_ensemble_v4.json'

# Run the consolidation function
majority_vote_consolidation_plaintext(file1, file2, file3, 'coco_ensemble_123.json')
majority_vote_consolidation_plaintext(file1, file2, file4, 'coco_ensemble_124.json')
majority_vote_consolidation_plaintext(file1, file2, file5, 'coco_ensemble_125.json')
majority_vote_consolidation_plaintext(file1, file3, file4, 'coco_ensemble_134.json')
majority_vote_consolidation_plaintext(file1, file3, file5, 'coco_ensemble_135.json')
majority_vote_consolidation_plaintext(file1, file4, file5, 'coco_ensemble_145.json')
majority_vote_consolidation_plaintext(file2, file3, file4, 'coco_ensemble_234.json')
majority_vote_consolidation_plaintext(file2, file3, file5, 'coco_ensemble_235.json')
majority_vote_consolidation_plaintext(file2, file4, file5, 'coco_ensemble_245.json')
majority_vote_consolidation_plaintext(file3, file4, file5, 'coco_ensemble_345.json')
