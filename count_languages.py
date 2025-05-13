import json

def count_language_tags(json_file_path):
    # Load the structured JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize counters
    language_counts = {}
    block_language_distribution = {}
    
    # Iterate through all blocks
    for block_id, block_data in data.items():
        if 'tokens' not in block_data:
            continue
            
        # Initialize block in distribution if not exists
        if block_id not in block_language_distribution:
            block_language_distribution[block_id] = {}
            
        # Iterate through sentences in the block
        for sentence_id, sentence_data in block_data['tokens'].items():
            if 'words' not in sentence_data:
                continue
                
            # Iterate through words in the sentence
            for word_id, word_data in sentence_data['words'].items():
                language = word_data.get('language', 'unknown')
                
                # Count language globally
                if language not in language_counts:
                    language_counts[language] = 0
                language_counts[language] += 1
                
                # Count language per block
                if language not in block_language_distribution[block_id]:
                    block_language_distribution[block_id][language] = 0
                block_language_distribution[block_id][language] += 1
    
    # Print summary
    print("Total language tag counts:")
    for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{lang}: {count}")
    
    print("\nLanguage distribution by block:")
    for block_id, lang_counts in block_language_distribution.items():
        print(f"\nBlock {block_id}:")
        for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {lang}: {count}")

if __name__ == "__main__":
    # Change this to your actual file path
    json_file_path = "translatable_structured.json"
    count_language_tags(json_file_path)
