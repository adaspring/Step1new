import json
import sys
from collections import Counter

def collect_languages(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "language":
                yield v
            else:
                yield from collect_languages(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from collect_languages(item)

def main():
    if len(sys.argv) != 2:
        print("Usage: python count_languages.py <input_file.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    global_counter = Counter()
    block_language_counts = {}

    for block_id in sorted(data.keys()):
        block = data[block_id]
        langs = list(collect_languages(block))
        counter = Counter(str(lang) for lang in langs)
        block_language_counts[block_id] = dict(counter)
        global_counter.update(counter)

    for block_id, counts in block_language_counts.items():
        print(f"{block_id}: {counts}")
    print("Global:", dict(global_counter))

if __name__ == "__main__":
    main()
