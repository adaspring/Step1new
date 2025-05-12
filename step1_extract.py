import os
import sys
import json
import uuid
import spacy
import argparse
import subprocess
import regex as re
from langdetect import detect
from pypinyin import lazy_pinyin
from bs4 import BeautifulSoup, Comment, NavigableString

# spaCy models mapped to language codes
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "de": "de_core_news_sm",
    "zh": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
    "ru": "ru_core_news_sm",
    "pt": "pt_core_news_sm",
    "it": "it_core_news_sm",
    "nl": "nl_core_news_sm",
    "nb": "nb_core_news_sm"
}

# Cache for loaded spaCy models
LOADED_SPACY_MODELS = {}

# HTML elements containing translatable text
TRANSLATABLE_TAGS = {
    "p", "span", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "label", "button", "li", "td", "th", "a", "strong", "em",
    "b", "i", "caption", "summary", "figcaption", "option", "optgroup",
    "legend", "mark", "output", "details", "time"
}

# HTML attributes containing translatable text
TRANSLATABLE_ATTRS = {
    "alt", "title", "placeholder", "aria-label", "aria-placeholder",
    "aria-valuetext", "aria-roledescription", "value",
    "data-i18n", "data-caption", "data-title", "data-tooltip",
    "data-label", "data-error"
}

# SEO meta fields that should be translated
SEO_META_FIELDS = {
    "name": {
        "description", "keywords", "robots", "author", "viewport", "theme-color"
    },
    "property": {
        "og:title", "og:description", "og:image", "og:url",
        "twitter:title", "twitter:description", "twitter:image", "twitter:card"
    }
}

# JSON-LD keys containing translatable content
TRANSLATABLE_JSONLD_KEYS = {
    "name", "description", "headline", "caption",
    "alternateName", "summary", "title", "about"
}

# Parent elements to skip during processing
SKIP_PARENTS = {
    "script", "style", "code", "pre", "noscript", "template", "svg", "canvas",
    "frameset", "frame", "noframes", "object", "embed", "base", "map"
}

# Attributes to exclude from translation
BLOCKED_ATTRS = {
    "accept", "align", "autocomplete", "bgcolor", "charset", "class", "content",
    "dir", "download", "href", "id", "lang", "name", "rel", "src", "style", "type"
}

# JSON-LD keys to exclude from processing
JSONLD_EXCLUDE_KEYS = {"duration", "uploadDate", "embedUrl", "contentUrl", "thumbnailUrl"}

# Excluded meta fields
EXCLUDED_META_NAMES = {"viewport"}
EXCLUDED_META_PROPERTIES = {"og:url"}

# Helper Functions -------------------------------------------------

def is_pure_symbol(text):
    """Check if text contains no alphabetic characters."""
    return not re.search(r'[A-Za-z]', text)

def detect_language(text):
    """Wrapper for langdetect with error handling."""
    try:
        return detect(text)
    except:
        return "unknown"

def is_symbol_heavy(text):
    """Determine if text contains mostly symbols/no real words."""
    words = re.findall(r'\b\p{L}{3,}\b', text)
    if len(words) > 0:
        return False
    symbol_count = len(re.findall(r'[\p{P}\p{S}\d_]', text))
    return symbol_count > 0

def is_exception_language(text):
    """Check for languages that should bypass symbol checks."""
    return contains_chinese(text) or re.search(r'[\u0600-\u06FF\u0400-\u04FF\u0370-\u03FF]', text)

def has_real_words(text):
    """Check for presence of real words (3+ letters)."""
    return re.search(r'\b\p{L}{3,}\b', text, re.UNICODE) is not None

def has_math_html_markup(element):
    """Check for math-related HTML markup in parent elements."""
    parent = element.parent
    return (
        parent.name == 'math' or 
        re.search(r'\$.*?\$|\\\(.*?\\\)', parent.text or '') or
        any(cls in parent.get('class', []) for cls in ['math', 'equation', 'formula'])
    )

def is_math_fragment(text):
    """Identify mathematical expressions in text."""
    equation_pattern = r'''
        (\w+\s*[=+\-*/^]\s*\S+)|  # Simple equations
        (\d+[\+\-\*/]\d+)|         # Arithmetic operations
        ([a-zA-Z]+\^?\d+)|          # Exponents
        (\$.*?\$|\\\(.*?\\\))       # LaTeX markup
    '''
    has_math = re.search(equation_pattern, text, re.VERBOSE)
    return (has_math and not has_real_words(text)) or is_symbol_heavy(text)

def get_spacy_model(lang_code):
    """Load or download spaCy model for specified language."""
    if lang_code not in SPACY_MODELS:
        return None

    if lang_code in LOADED_SPACY_MODELS:
        return LOADED_SPACY_MODELS[lang_code]

    model_name = SPACY_MODELS[lang_code]
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Downloading missing spaCy model '{model_name}'...")
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        nlp = spacy.load(model_name)

    LOADED_SPACY_MODELS[lang_code] = nlp
    return nlp

def is_translatable_text(tag):
    """Determine if element's text should be translated."""
    # Check translate attribute inheritance
    current_element = tag.parent
    translate_override = None
    
    while current_element is not None:
        current_translate = current_element.get("translate", "").lower()
        if current_translate in {"yes", "no"}:
            translate_override = current_translate
            break
        current_element = current_element.parent

    text = tag.strip()
    if not text:
        return False

    # Skip math/symbol content except for certain languages
    if ((not is_exception_language(text)) 
    and (
        is_pure_symbol(text) or 
        is_math_fragment(text) or 
        has_math_html_markup(tag))):
        return False

    if translate_override == "no":
        return False

    parent_tag = tag.parent.name if tag.parent else None
    default_translatable = (
        parent_tag in TRANSLATABLE_TAGS and
        parent_tag not in SKIP_PARENTS and
        not isinstance(tag, Comment))
        
    return default_translatable if translate_override is None else (translate_override == "yes")

def contains_chinese(text):
    """Check for presence of Chinese characters."""
    return re.search(r'[\u4e00-\u9fff]', text) is not None

# Core Processing Functions ----------------------------------------

def process_text_with_language_detection(block_id, text):
    """Process text using language-specific spaCy model."""
    try:
        detected_lang = detect(text)
        lang_mapping = {
            "en": "en", "fr": "fr", "es": "es", "de": "de",
            "zh-cn": "zh", "zh-tw": "zh", "zh": "zh",
            "ja": "ja", "ko": "ko", "ru": "ru", "pt": "pt",
            "it": "it", "nl": "nl", "no": "nb"
        }
        spacy_lang = lang_mapping.get(detected_lang, "en")
        if spacy_lang not in SPACY_MODELS:
            spacy_lang = "en"
        nlp = get_spacy_model(spacy_lang)
        print(f"Processing block {block_id} ({detected_lang} → {spacy_lang} model)")
    except Exception as e:
        print(f"Language detection failed: {e}. Using English model.")
        nlp = get_spacy_model("en")
    return process_text_block(block_id, text, nlp)

def process_text_block(block_id, text, nlp):
    """Process text into structured linguistic components."""
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
            structured[s_key]["words"][w_key] = {
                "text": token.text,
                "pos": token.pos_,
                "ent": token.ent_type_ or None,
                "pinyin": " ".join(lazy_pinyin(token.text)) if contains_chinese(token.text) else None
            }

    return structured, flattened, sentence_tokens

def extract_from_jsonld(obj, block_counter, structured_output, flattened_output):
    """Recursively extract translatable content from JSON-LD data."""
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            value = obj[key]
            if isinstance(value, str):
                key_lc = key.lower()
                if (key_lc not in JSONLD_EXCLUDE_KEYS and 
                    (key_lc in TRANSLATABLE_JSONLD_KEYS or 
                     (not key_lc.startswith("@") and
                      all(x not in key_lc for x in ["url", "date", "time", "type"]))
                   ):
                    block_id = f"BLOCK_{block_counter}"
                    structured, flattened, tokens = process_text_with_language_detection(block_id, value)
                    obj[key] = tokens[0][0]
                    structured_output[block_id] = {"jsonld": key, "tokens": structured}
                    flattened_output.update(flattened)
                    block_counter += 1
            elif isinstance(value, (dict, list)):
                block_counter = extract_from_jsonld(value, block_counter, structured_output, flattened_output)
    elif isinstance(obj, list):  # Removed extra parenthesis here
        for i in range(len(obj)):
            block_counter = extract_from_jsonld(obj[i], block_counter, structured_output, flattened_output)
    return block_counter

def extract_translatable_html(input_path):
    """Main extraction function with per-block language detection."""
    with open(input_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html5lib")

    structured_output = {}
    flattened_output = {}
    block_counter = 1

    # Process text nodes
    elements = list(soup.find_all(string=True))
    for element in elements:
        if is_translatable_text(element):
            text = element.strip()
            if text:
                structured, flattened, sentence_tokens = process_text_with_language_detection(
                    f"BLOCK_{block_counter}", text)
                
                if sentence_tokens:
                    block_id = f"BLOCK_{block_counter}"
                    parent_tag = element.parent.name if element.parent else "no_parent"
                    structured_output[block_id] = {"tag": parent_tag, "tokens": structured}
                    flattened_output.update(flattened)
                    
                    replacement_content = sentence_tokens[0][0]
                    if not isinstance(replacement_content, NavigableString):
                        replacement_content = NavigableString(str(replacement_content))
                    element.replace_with(replacement_content)
                    
                    block_counter += 1

    # Process attributes
    for tag in soup.find_all():
        for attr in TRANSLATABLE_ATTRS:
            if (attr in tag.attrs and 
                isinstance(tag[attr], str) and 
                attr not in BLOCKED_ATTRS):
                value = tag[attr].strip()
                if value:
                    block_id = f"BLOCK_{block_counter}"
                    structured, flattened, sentence_tokens = process_text_with_language_detection(block_id, value)
                    structured_output[block_id] = {"attr": attr, "tokens": structured}
                    flattened_output.update(flattened)
                    if sentence_tokens:
                        tag[attr] = sentence_tokens[0][0]
                    block_counter += 1

    # Process meta tags
    for meta in soup.find_all("meta"):
        name = meta.get("name", "").lower()
        prop = meta.get("property", "").lower()
        content = meta.get("content", "").strip()

        if name in EXCLUDED_META_NAMES or prop in EXCLUDED_META_PROPERTIES:
            continue

        if content and ((name in SEO_META_FIELDS["name"]) or (prop in SEO_META_FIELDS["property"])):
            block_id = f"BLOCK_{block_counter}"
            structured, flattened, sentence_tokens = process_text_with_language_detection(block_id, content)
            structured_output[block_id] = {"meta": name or prop, "tokens": structured}
            flattened_output.update(flattened)
            if sentence_tokens:
                meta["content"] = sentence_tokens[0][0]
            block_counter += 1

    # Process title tag
    title_tag = soup.title
    if title_tag and title_tag.string and title_tag.string.strip():
        block_id = f"BLOCK_{block_counter}"
        structured, flattened, sentence_tokens = process_text_with_language_detection(block_id, title_tag.string.strip())
        structured_output[block_id] = {"tag": "title", "tokens": structured}
        flattened_output.update(flattened)
        if sentence_tokens:
            title_tag.string.replace_with(sentence_tokens[0][0])
        block_counter += 1

    # Process JSON-LD data
    for script_tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(script_tag.string.strip())
            block_counter = extract_from_jsonld(data, block_counter, structured_output, flattened_output)
            script_tag.string.replace_with(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"⚠️ Failed to process JSON-LD: {e}")

    # Save output files
    with open("translatable_flat.json", "w", encoding="utf-8") as f:
        json.dump(flattened_output, f, indent=2, ensure_ascii=False)

    with open("translatable_structured.json", "w", encoding="utf-8") as f:
        json.dump(structured_output, f, indent=2, ensure_ascii=False)

    with open("non_translatable.html", "w", encoding="utf-8") as f:
        f.write(str(soup))

    # Create sentences-only version
    flat_sentences_only = {k: v for k, v in flattened_output.items() if "_S" in k and "_W" not in k}
    with open("translatable_flat_sentences.json", "w", encoding="utf-8") as f:
        json.dump(flat_sentences_only, f, indent=2, ensure_ascii=False)

    print("✅ Extraction complete. Output files created:\n"
          "- translatable_flat.json\n"
          "- translatable_structured.json\n"
          "- non_translatable.html\n"
          "- translatable_flat_sentences.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract translatable content from HTML with language detection")
    parser.add_argument("input_file", help="Path to input HTML file")
    args = parser.parse_args()
    extract_translatable_html(args.input_file)
