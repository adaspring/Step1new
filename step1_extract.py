import os
import sys
import json
import uuid
import spacy
import argparse
import subprocess
from bs4 import BeautifulSoup, Comment, NavigableString

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

TRANSLATABLE_JSONLD_KEYS = {
    "name", "description", "headline", "caption",
    "alternateName", "summary", "title", "about"
}

SKIP_PARENTS = {
    "script", "style", "code", "pre", "noscript", "template", "svg", "canvas",
    "frameset", "frame", "noframes", "object", "embed", "base", "map"
}

BLOCKED_ATTRS = {
    "accept", "align", "autocomplete", "bgcolor", "charset", "class", "content",
    "dir", "download", "href", "id", "lang", "name", "rel", "src", "style", "type"
}

JSONLD_EXCLUDE_KEYS = {
    "duration", "uploadDate", "embedUrl", "contentUrl", "thumbnailUrl"
}

EXCLUDED_META_NAMES = {"viewport"}
EXCLUDED_META_PROPERTIES = {"og:url"}


# Helper Functions -------------------------------------------------
def is_pure_symbol(text):
    """Skip text with no alphabetic characters."""
    return not re.search(r'[A-Za-z]', text)

def has_math_html_markup(element):
    """Check for math-specific HTML markup (MathML, LaTeX, etc.)."""
    parent = element.parent
    return (
        parent.name == 'math' or 
        re.search(r'\$.*?\$|\\\(.*?\\\)', parent.text) or
        any(cls in parent.get('class', []) for cls in ['math', 'equation', 'formula'])
    )

def is_math_fragment(text):
    """Check if text is a math formula without lexical words."""
    equation_pattern = r'''
        (\w+\s*[=+\-*/^]\s*\S+)|  # Equations like "x = y+1"
        (\d+[\+\-\*/]\d+)|         # Arithmetic "2+3"
        ([a-zA-Z]+\^?\d+)|         # Exponents "x²"
        (\$.*?\$|\\\(.*?\\\))      # LaTeX "$E=mc^2$"
    '''
    has_math = re.search(equation_pattern, text, re.VERBOSE)
    has_lexical = re.search(r'[A-Za-z]', text)
    return has_math and not has_lexical

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
    # Check translate attribute inheritance hierarchy
    current_element = tag.parent
    translate_override = None
    
    while current_element is not None:
        current_translate = current_element.get("translate", "").lower()
        
        if current_translate in {"yes", "no"}:
            translate_override = current_translate
            break  # Closest explicit declaration wins
        current_element = current_element.parent

    # If any parent says "no", block translation
    if translate_override == "no":
        return False

    # If no explicit "yes", check default translatability
    default_translatable = (
        tag.parent.name in TRANSLATABLE_TAGS and
        tag.parent.name not in SKIP_PARENTS and
        not isinstance(tag, Comment) and
        tag.strip()
    )

    # Explicit "yes" overrides default logic
    if translate_override == "yes":
        return default_translatable or True  # Force allow if parent says "yes"
        
    return default_translatable

# Updated Translatability Check -------------------------------------
def is_translatable_text(element):
    text = element.strip()
    if not text:
        return False

    # Skip parents like <script>, <style>, etc.
    if element.parent.name in SKIP_PARENTS:
        return False

    # Skip pure symbols (e.g., ">", "***")
    if is_pure_symbol(text):
        return False

    # Skip math fragments or HTML-marked math
    if is_math_fragment(text) or has_math_html_markup(element):
        return False

    # Original logic for translatability (check parents/tags)
    return (
        element.parent.name in TRANSLATABLE_TAGS and
        not isinstance(element, Comment) and
        element.parent.name not in SKIP_PARENTS
    )







def process_text_block(block_id, text, nlp):
    structured = {}
    flattened = {}
    sentence_tokens = []

    doc = nlp(text)
    for s_idx, sent in enumerate(doc.sents, 1):
        s_key = f"S{s_idx}"
        sentence_id = f"{block_id}_{s_key}"
        sentence_text = sent.text
        flattened[sentence_id] = sentence_text
        structured[s_key] = {"text": sentence_text, "words": {}}
        sentence_tokens.append((sentence_id, sentence_text))

        for w_idx, token in enumerate(sent, 1):
            w_key = f"W{w_idx}"
            word_id = f"{sentence_id}_{w_key}"
            flattened[word_id] = token.text
            structured[s_key]["words"][w_key] = token.text

    return structured, flattened, sentence_tokens


def extract_from_jsonld(obj, block_counter, nlp, structured_output, flattened_output):
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            value = obj[key]
            if isinstance(value, str):
                key_lc = key.lower()
                if (
                    key_lc not in JSONLD_EXCLUDE_KEYS and (
                        key_lc in TRANSLATABLE_JSONLD_KEYS or (
                            not key_lc.startswith("@") and
                            all(x not in key_lc for x in ["url", "date", "time", "type"])
                        )
                    )
                ):
                    block_id = f"BLOCK_{block_counter}"
                    structured, flattened, tokens = process_text_block(block_id, value, nlp)
                    obj[key] = tokens[0][0]
                    structured_output[block_id] = {"jsonld": key, "tokens": structured}
                    flattened_output.update(flattened)
                    block_counter += 1
            elif isinstance(value, (dict, list)):
                block_counter = extract_from_jsonld(value, block_counter, nlp, structured_output, flattened_output)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            block_counter = extract_from_jsonld(obj[i], block_counter, nlp, structured_output, flattened_output)
    return block_counter


def extract_translatable_html(input_path, lang_code):
    nlp = load_spacy_model(lang_code)

    with open(input_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html5lib")

    structured_output = {}
    flattened_output = {}
    block_counter = 1

    elements = list(soup.find_all(string=True))  # Fix 1: Precompute elements
    for element in elements:
        if is_translatable_text(element):
            text = element.strip()
            if not text:
                continue

            structured, flattened, sentence_tokens = process_text_block(f"BLOCK_{block_counter}", text, nlp)

            if sentence_tokens:
                block_id = f"BLOCK_{block_counter}"
                parent_tag = element.parent.name if element.parent else "no_parent"  # Fix 2: Parent check
                structured_output[block_id] = {"tag": parent_tag, "tokens": structured}
                flattened_output.update(flattened)
                
                # Fix 3: Safe replacement
                replacement_content = sentence_tokens[0][0]
                if not isinstance(replacement_content, NavigableString):
                    replacement_content = NavigableString(str(replacement_content))
                element.replace_with(replacement_content)
                
                block_counter += 1

    for tag in soup.find_all():
        for attr in TRANSLATABLE_ATTRS:
            if (
                attr in tag.attrs and 
                isinstance(tag[attr], str) and 
                attr not in BLOCKED_ATTRS
            ):
                value = tag[attr].strip()
                if value:
                    block_id = f"BLOCK_{block_counter}"
                    structured, flattened, sentence_tokens = process_text_block(block_id, value, nlp)
                    structured_output[block_id] = {"attr": attr, "tokens": structured}
                    flattened_output.update(flattened)
                    if sentence_tokens:
                        tag[attr] = sentence_tokens[0][0]
                    block_counter += 1

    for meta in soup.find_all("meta"):
        name = meta.get("name", "").lower()
        prop = meta.get("property", "").lower()
        content = meta.get("content", "").strip()

        if name in EXCLUDED_META_NAMES or prop in EXCLUDED_META_PROPERTIES:
            continue

        if content and (
            (name and name in SEO_META_FIELDS["name"]) or
            (prop and prop in SEO_META_FIELDS["property"])
        ):
            block_id = f"BLOCK_{block_counter}"
            structured, flattened, sentence_tokens = process_text_block(block_id, content, nlp)
            structured_output[block_id] = {"meta": name or prop, "tokens": structured}
            flattened_output.update(flattened)
            if sentence_tokens:
                meta["content"] = sentence_tokens[0][0]
            block_counter += 1

    title_tag = soup.title
    if title_tag and title_tag.string and title_tag.string.strip():
        block_id = f"BLOCK_{block_counter}"
        text = title_tag.string.strip()
        structured, flattened, sentence_tokens = process_text_block(block_id, text, nlp)
        structured_output[block_id] = {"tag": "title", "tokens": structured}
        flattened_output.update(flattened)
        if sentence_tokens:
            title_tag.string.replace_with(sentence_tokens[0][0])
        block_counter += 1

    for script_tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            raw_json = script_tag.string.strip()
            data = json.loads(raw_json)
            block_counter = extract_from_jsonld(data, block_counter, nlp, structured_output, flattened_output)
            script_tag.string.replace_with(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"⚠️ Failed to parse or process JSON-LD: {e}")
            continue

    with open("translatable_flat.json", "w", encoding="utf-8") as f:
        json.dump(flattened_output, f, indent=2, ensure_ascii=False)

    with open("translatable_structured.json", "w", encoding="utf-8") as f:
        json.dump(structured_output, f, indent=2, ensure_ascii=False)

    with open("non_translatable.html", "w", encoding="utf-8") as f:
        f.write(str(soup))

    flat_sentences_only = {
        k: v for k, v in flattened_output.items()
        if "_S" in k and "_W" not in k
    }
    with open("translatable_flat_sentences.json", "w", encoding="utf-8") as f:
        json.dump(flat_sentences_only, f, indent=2, ensure_ascii=False)

    print("✅ Step 1 complete: saved translatable_flat.json, translatable_structured.json, and non_translatable.html.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="HTML file to process")
    parser.add_argument("--lang", choices=SPACY_MODELS.keys(), default="en", help="Language code (default: en)")
    args = parser.parse_args()
    extract_translatable_html(args.input_file, args.lang)
