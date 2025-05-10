import os
import sys
import json
import uuid
import spacy
import argparse
import subprocess
from bs4 import BeautifulSoup, Comment

SPACY_MODELS = {
    "en": "en_core_web_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "de": "de_core_news_sm"
}

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
    "name": {
        "description", "keywords", "robots", "author", "viewport", "theme-color"
    },
    "property": {
        "og:title", "og:description", "og:image", "og:url",
        "twitter:title", "twitter:description", "twitter:image", "twitter:card"
    }
}

SKIP_PARENTS = {
    "script", "style", "code", "pre", "noscript", "template", "svg", "canvas"
}

def load_spacy_model(lang_code):
    if lang_code not in SPACY_MODELS:
        print(f"Unsupported language '{lang_code}'. Choose from: {', '.join(SPACY_MODELS)}.")
        sys.exit(1)

    model_name = SPACY_MODELS[lang_code]

    try:
        return spacy.load(model_name)
    except OSError:
        print(f"spaCy model '{model_name}' not found. Downloading automatically...")
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        return spacy.load(model_name)

def is_translatable_text(tag):
    return (
        tag.parent.name in TRANSLATABLE_TAGS and
        tag.parent.name not in SKIP_PARENTS and
        not isinstance(tag, Comment) and
        tag.strip()
    )

def generate_token():
    return f"__TRANS_{uuid.uuid4().hex}__"

def process_text_block(block_id, text, nlp):
    structured = {}
    flattened = {}

    doc = nlp(text)
    for s_idx, sent in enumerate(doc.sents, 1):
        s_key = f"S{s_idx}"
        flattened[f"{block_id}_{s_key}"] = sent.text
        structured[s_key] = {"text": sent.text, "words": {}}

        for w_idx, token in enumerate(sent, 1):
            w_key = f"W{w_idx}"
            word_key = f"{block_id}_{s_key}_{w_key}"
            flattened[word_key] = token.text
            structured[s_key]["words"][w_key] = token.text

    return structured, flattened

def extract_translatable_html(input_path, lang_code):
    nlp = load_spacy_model(lang_code)

    with open(input_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    structured_output = {}
    flattened_output = {}
    block_counter = 1

    for element in soup.find_all(string=True):
        if is_translatable_text(element):
            text = element.strip()
            block_id = f"BLOCK_{block_counter}"
            structured, flattened = process_text_block(block_id, text, nlp)
            structured_output[block_id] = {
                "tag": element.parent.name,
                "tokens": structured
            }
            flattened_output.update(flattened)

            token = generate_token()
            element.replace_with(token)
            block_counter += 1

    for tag in soup.find_all():
        for attr in TRANSLATABLE_ATTRS:
            if attr in tag.attrs and isinstance(tag[attr], str):
                value = tag[attr].strip()
                if value:
                    block_id = f"BLOCK_{block_counter}"
                    structured, flattened = process_text_block(block_id, value, nlp)
                    structured_output[block_id] = {"attr": attr, "tokens": structured}
                    flattened_output.update(flattened)

                    token = generate_token()
                    tag[attr] = token
                    block_counter += 1

    for meta in soup.find_all("meta"):
        name = meta.get("name", "").lower()
        prop = meta.get("property", "").lower()
        content = meta.get("content", "").strip()
        if content and (name in SEO_META_FIELDS["name"] or prop in SEO_META_FIELDS["property"]):
            block_id = f"BLOCK_{block_counter}"
            structured, flattened = process_text_block(block_id, content, nlp)
            structured_output[block_id] = {"meta": name or prop, "tokens": structured}
            flattened_output.update(flattened)

            token = generate_token()
            meta["content"] = token
            block_counter += 1

    title_tag = soup.title
    if title_tag and title_tag.string and title_tag.string.strip():
        block_id = f"BLOCK_{block_counter}"
        text = title_tag.string.strip()
        structured, flattened = process_text_block(block_id, text, nlp)
        structured_output[block_id] = {"tag": "title", "tokens": structured}
        flattened_output.update(flattened)

        token = generate_token()
        title_tag.string.replace_with(token)
        block_counter += 1

    with open("translatable_flat.json", "w", encoding="utf-8") as f:
        json.dump(flattened_output, f, indent=2, ensure_ascii=False)

    with open("translatable_structured.json", "w", encoding="utf-8") as f:
        json.dump(structured_output, f, indent=2, ensure_ascii=False)

    with open("non_translatable.html", "w", encoding="utf-8") as f:
        f.write(str(soup))

    print("✅ Step 1 complete: saved translatable_flat.json, translatable_structured.json, and non_translatable.html.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="HTML file to process")
    parser.add_argument("--lang", choices=SPACY_MODELS.keys(), default="en", help="Language code (default: en)")
    args = parser.parse_args()

    extract_translatable_html(args.input_file, args.lang)
