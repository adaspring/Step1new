import json
from collections import defaultdict, Counter
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python count_languages.py <input_file.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    block_language_counts = {}
    global_counter = Counter()

    for block_id in sorted(data.keys()):
        block = data[block_id]
        word_langs = [word_data["language"] for word_data in block["words"].values()]
        count = Counter(word_langs)
        block_language_counts[block_id] = dict(count)
        global_counter.update(count)

    for block_id, counts in block_language_counts.items():
        print(f"{block_id}: {counts}")

    print("Global:", dict(global_counter))

if __name__ == "__main__":
    main()
