import os
import sys
import json
import uuid
import spacy
from bs4 import BeautifulSoup, Comment

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define translatable tags and attributes
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

def flatten_sentence_tokens(text, block_id, sentence_index):
    flat_map = {}
    sentence_id = f"{block_id}_S{sentence_index}"
    flat_map[sentence_id] = text
    doc = nlp(text)
    for i, token in enumerate(doc):
        if not token.is_space:
            word_key = f"{sentence_id}_W{i+1}"
            flat_map[word_key] = token.text
    return flat_map

def extract_translatable_html(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    flat_token_map = {}
    block_count = 0

    # Text nodes in elements
    for element in soup.find_all(string=True):
        if is_translatable_text(element):
            block_count += 1
            block_id = f"BLOCK_{block_count}"
            doc = nlp(element.strip())
            sentence_index = 1
            for sent in doc.sents:
                sentence = sent.text.strip()
                if sentence:
                    flat_map = flatten_sentence_tokens(sentence, block_id, sentence_index)
                    flat_token_map.update(flat_map)
                    sentence_index += 1
            # Replace element with tokens
            element.replace_with(" ".join([f"__{block_id}_S{i}__" for i in range(1, sentence_index)]))

    # Translatable attributes
    for tag in soup.find_all():
        for attr in TRANSLATABLE_ATTRS:
            if attr in tag.attrs and isinstance(tag[attr], str):
                value = tag[attr].strip()
                if value:
                    block_count += 1
                    block_id = f"BLOCK_{block_count}"
                    doc = nlp(value)
                    sentence_index = 1
                    tokens = []
                    for sent in doc.sents:
                        sentence = sent.text.strip()
                        if sentence:
                            flat_map = flatten_sentence_tokens(sentence, block_id, sentence_index)
                            flat_token_map.update(flat_map)
                            tokens.append(f"__{block_id}_S{sentence_index}__")
                            sentence_index += 1
                    tag[attr] = " ".join(tokens)

    # SEO meta content
    for meta in soup.find_all("meta"):
        name = meta.get("name", "").lower()
        prop = meta.get("property", "").lower()
        content = meta.get("content", "").strip()
        if content and (name in SEO_META_FIELDS["name"] or prop in SEO_META_FIELDS["property"]):
            block_count += 1
            block_id = f"BLOCK_{block_count}"
            doc = nlp(content)
            sentence_index = 1
            tokens = []
            for sent in doc.sents:
                sentence = sent.text.strip()
                if sentence:
                    flat_map = flatten_sentence_tokens(sentence, block_id, sentence_index)
                    flat_token_map.update(flat_map)
                    tokens.append(f"__{block_id}_S{sentence_index}__")
                    sentence_index += 1
            meta["content"] = " ".join(tokens)

    # Title
    title_tag = soup.title
    if title_tag and title_tag.string and title_tag.string.strip():
        block_count += 1
        block_id = f"BLOCK_{block_count}"
        doc = nlp(title_tag.string.strip())
        sentence_index = 1
        tokens = []
        for sent in doc.sents:
            sentence = sent.text.strip()
            if sentence:
                flat_map = flatten_sentence_tokens(sentence, block_id, sentence_index)
                flat_token_map.update(flat_map)
                tokens.append(f"__{block_id}_S{sentence_index}__")
                sentence_index += 1
        title_tag.string.replace_with(" ".join(tokens))

    with open("translatable_flat.json", "w", encoding="utf-8") as f:
        json.dump(flat_token_map, f, indent=2, ensure_ascii=False)

    with open("non_translatable.html", "w", encoding="utf-8") as f:
        f.write(str(soup))

    print("✅ Step 1 complete: created translatable_flat.json and non_translatable.html")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python translate_extract_step1_flattened.py <input_file>")
        sys.exit(1)

    extract_translatable_html(sys.argv[1])
