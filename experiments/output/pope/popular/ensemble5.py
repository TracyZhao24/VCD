import json
from collections import Counter

def majority_vote_consolidation_plaintext(file1, file2, file3, file4, file5, output_file):
    """
    Consolidates five plaintext files (line-delimited JSON) based on majority voting for the 'text' field.

    Args:
    file1, file2, file3, file4, file5: Input plaintext files containing JSON entries.
    output_file: Path to the output plaintext file.
    """
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(file3, 'r') as f3, open(file4, 'r') as f4, open(file5, 'r') as f5:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        lines3 = f3.readlines()
        lines4 = f4.readlines()
        lines5 = f5.readlines()

    # Ensure all files have the same number of lines
    assert len(lines1) == len(lines2) == len(lines3) == len(lines4) == len(lines5), "Files must have the same number of entries."

    consolidated_data = []

    for line1, line2, line3, line4, line5 in zip(lines1, lines2, lines3, lines4, lines5):
        entry1 = json.loads(line1.strip())
        entry2 = json.loads(line2.strip())
        entry3 = json.loads(line3.strip())
        entry4 = json.loads(line4.strip())
        entry5 = json.loads(line5.strip())

        assert entry1["question_id"] == entry2["question_id"] == entry3["question_id"] == entry4["question_id"] == entry5["question_id"], \
            "Question IDs do not match between files."

        # Gather the 'text' votes
        votes = [entry1["text"], entry2["text"], entry3["text"], entry4["text"], entry5["text"]]

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
output_file = 'coco_ensemble5_popular.json'

# Run the consolidation function
majority_vote_consolidation_plaintext(file1, file2, file3, file4, file5, output_file)