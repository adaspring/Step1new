import os
import sys
import json
import spacy
from bs4 import BeautifulSoup, Comment

nlp = spacy.load("en_core_web_sm")

TRANSLATABLE_TAGS = {
    "p", "span", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "label", "button", "li", "td", "th", "a", "strong", "em",
    "b", "i", "caption", "summary", "figcaption", "option", "optgroup",
    "legend", "mark", "output", "details", "time"
}

TRANSLATABLE_ATTRS = {
    "alt", "title", "placeholder", "aria-label", "aria-placeholder",
    "aria-valuetext", "aria-roledescription", "value",
    "data-i18n", "data-caption", "data-title", "data-tooltip",
    "data-label", "data-error"
}

SEO_META_FIELDS = {
    "name": {"description", "keywords", "robots", "author", "viewport", "theme-color"},
    "property": {
        "og:title", "og:description", "og:image", "og:url",
        "twitter:title", "twitter:description", "twitter:image", "twitter:card"
    }
}

SKIP_PARENTS = {"script", "style", "code", "pre", "noscript", "template", "svg", "canvas"}

def is_translatable_text(tag):
    return (
        tag.parent.name in TRANSLATABLE_TAGS and
        tag.parent.name not in SKIP_PARENTS and
        not isinstance(tag, Comment) and
        tag.strip()
    )

def is_lexical_content(text):
    doc = nlp(text)
    return any(token.is_alpha or token.is_digit for token in doc if not token.is_space)

def flatten_and_structure(text, block_id, sentence_index):
    flat_map = {}
    structured = {}

    sentence_id = f"{block_id}_S{sentence_index}"
    flat_map[sentence_id] = text
    doc = nlp(text)

    tokens = [token for token in doc if not token.is_space]
    joined = " ".join([token.text for token in tokens])
    if len(tokens) <= 3 and joined == text:
        structured["text"] = text
        structured["words"] = [{"id": "W1", "text": text}]
        return flat_map, structured

    structured["text"] = text
    structured["words"] = []
    for i, token in enumerate(tokens):
        word_id = f"W{i+1}"
        flat_map[f"{sentence_id}_{word_id}"] = token.text
        structured["words"].append({
            "id": word_id,
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dep": token.dep_
        })

    return flat_map, structured

def process_json_ld(obj, block_id, flat_token_map, structured_map, sentence_index=1):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key], sentence_index = process_json_ld(value, block_id, flat_token_map, structured_map, sentence_index)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i], sentence_index = process_json_ld(obj[i], block_id, flat_token_map, structured_map, sentence_index)
    elif isinstance(obj, str) and is_lexical_content(obj):
        flat_map, structured = flatten_and_structure(obj, block_id, sentence_index)
        flat_token_map.update(flat_map)
        if block_id not in structured_map:
            structured_map[block_id] = {"tag": "script[type=application/ld+json]", "sentences": {}}
        structured_map[block_id]["sentences"][f"S{sentence_index}"] = structured
        return f"__{block_id}_S{sentence_index}__", sentence_index + 1
    return obj, sentence_index

def extract_translatable_html(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    flat_token_map = {}
    structured_map = {}
    block_count = 0

    # 1. Extract normal visible text
    for element in soup.find_all(string=True):
        if is_translatable_text(element):
            doc = nlp(element.strip())
            sentences = [sent.text.strip() for sent in doc.sents if is_lexical_content(sent.text.strip())]
            if not sentences:
                continue

            sentence_index = 1
            token_placeholders = []
            temp_flat_map = {}
            temp_structured = {}
            for sentence in sentences:
                flat_map, structured = flatten_and_structure(sentence, f"BLOCK_{block_count+1}", sentence_index)
                temp_flat_map.update(flat_map)
                temp_structured[f"S{sentence_index}"] = structured
                token_placeholders.append(f"__BLOCK_{block_count+1}_S{sentence_index}__")
                sentence_index += 1

            if token_placeholders:
                block_count += 1
                block_id = f"BLOCK_{block_count}"
                flat_token_map.update(temp_flat_map)
                structured_map[block_id] = {
                    "tag": element.parent.name,
                    "sentences": temp_structured
                }
                element.replace_with(" ".join(token_placeholders))

    # 2. Extract from application/ld+json blocks
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            json_data = json.loads(script.string)
        except Exception:
            continue  # skip if malformed
        block_count += 1
        block_id = f"BLOCK_{block_count}"
        processed_data, _ = process_json_ld(json_data, block_id, flat_token_map, structured_map)
        script.string.replace_with(json.dumps(processed_data, ensure_ascii=False, indent=2))
    
    # 3. Write output files
    with open("translatable_flat.json", "w", encoding="utf-8") as f:
        json.dump(flat_token_map, f, indent=2, ensure_ascii=False)

    with open("translatable_structured.json", "w", encoding="utf-8") as f:
        json.dump(structured_map, f, indent=2, ensure_ascii=False)

    with open("non_translatable.html", "w", encoding="utf-8") as f:
        f.write(str(soup))

    print("✅ Step 1 complete: extracted text, including from JSON-LD.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python step1_extract.py <input_file>")
        sys.exit(1)

    extract_translatable_html(sys.argv[1])
