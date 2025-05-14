import os
import sys
import json
import uuid
import spacy
import argparse
import subprocess
import regex as re
from pypinyin import lazy_pinyin
from bs4 import BeautifulSoup, Comment, NavigableString


SPACY_MODELS = {
    "en": "en_core_web_sm",
    "zh": "zh_core_web_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "de": "de_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "ru": "ru_core_news_sm",
    "el": "el_core_news_sm",
    "xx": "xx_ent_wiki_sm"  # fallback for multilingual
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

def is_symbol_heavy(text):
    """Skip only if there's zero real words and many symbols (multilingual safe)."""

    # Count real words of 3+ letters
    words = re.findall(r'\b\p{L}{3,}\b', text)
    word_count = len(words)

    # If there's at least one real word, it's not symbol-heavy
    if word_count > 0:
        return False

    # Otherwise check for excessive symbols
    symbol_count = len(re.findall(r'[\p{P}\p{S}\d_]', text))
    return symbol_count > 0  # treat as symbol-heavy if only symbols

def is_exception_language(text):
    """
    Detect if the text contains a script or pattern matching a non-default language.

    Returns:
        A language code (e.g. 'zh', 'fr', 'ru', 'xx') if a match is found.
        Returns None if no exception language is detected.
    """
    if contains_chinese(text):
        return "zh"
    elif contains_arabic(text):
        return "xx"
    elif contains_hebrew(text):
        return "xx"
    elif contains_thai(text):
        return "xx"
    elif contains_devanagari(text):
        return "xx"
    return None

def detectis_exception_language(text):
    """
    Detect if the text contains a script or pattern matching a non-default language.

    Returns:
        A language code (e.g. 'zh', 'fr', 'ru', 'xx') if a match is found.
        Returns None if no exception language is detected.
    """
    if contains_chinese(text):
        return "zh"
    elif contains_english(text):
        return "en"
    elif contains_arabic(text):
        return "xx"
    elif contains_cyrillic(text):
        return "ru"
    elif contains_greek(text):
        return "el"
    elif contains_hebrew(text):
        return "xx"
    elif contains_thai(text):
        return "xx"
    elif contains_devanagari(text):
        return "xx"
    elif contains_french(text):
        return "fr"
    elif contains_spanish(text):
        return "es"
    elif contains_italian(text):
        return "it"
    elif contains_german(text):
        return "de"
    return None


def has_real_words(text):
    return re.search(r'\b\p{L}{3,}\b', text, re.UNICODE) is not None

def has_math_html_markup(element):
    """Check for math-specific HTML markup (MathML, LaTeX, etc.)."""
    parent = element.parent
    return (
        parent.name == 'math' or 
        re.search(r'\$.*?\$|\\\(.*?\\\)', parent.text or '') or
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
    return (has_math and not has_real_words(text)) or is_symbol_heavy(text)  # <-- Fixed line continuation


def load_spacy_model(lang_code):
    if lang_code not in SPACY_MODELS:
        print(f"Unsupported language '{lang_code}'. Choose from: {', '.join(SPACY_MODELS)}.")
        sys.exit(1)

    model_name = SPACY_MODELS[lang_code]

    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"spaCy model '{model_name}' not found. Downloading automatically...")
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        nlp = spacy.load(model_name)

    # Minimal addition: ensure sentence segmentation
    if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)

    return nlp

def is_translatable_text(tag):
    """Determine if the given tag's text should be translated."""
    # Check translate attribute inheritance hierarchy
    current_element = tag.parent
    translate_override = None
    
    while current_element is not None:
        current_translate = current_element.get("translate", "").lower()
        if current_translate in {"yes", "no"}:
            translate_override = current_translate
            break  # Closest explicit declaration wins
        current_element = current_element.parent

    # Check text content after parent checks
    text = tag.strip()
    if not text:
        return False

    # Math and symbol skipping (with proper line continuation)
    if ((not is_exception_language(text)) 
    and (
        is_pure_symbol(text) or 
        is_math_fragment(text) or 
        has_math_html_markup(tag))):
        return False

    # If any parent says "no", block translation
    if translate_override == "no":
        return False

    # If no explicit "yes", check default translatability
    parent_tag = tag.parent.name if tag.parent else None
    default_translatable = (
        parent_tag in TRANSLATABLE_TAGS and
        parent_tag not in SKIP_PARENTS and
        not isinstance(tag, Comment))
    # Explicit "yes" overrides default logic
    if translate_override == "yes":
        return True  # Force allow if parent says "yes"
        
    return default_translatable



def contains_chinese(text):
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def contains_arabic(text):
    return re.search(r'[\u0600-\u06FF]', text) is not None

def contains_cyrillic(text):
    return re.search(r'[\u0400-\u04FF]', text) is not None

def contains_greek(text):
    return re.search(r'[\u0370-\u03FF]', text) is not None

def contains_hebrew(text):
    return re.search(r'[\u0590-\u05FF]', text) is not None

def contains_thai(text):
    return re.search(r'[\u0E00-\u0E7F]', text) is not None

def contains_devanagari(text):
    return re.search(r'[\u0900-\u097F]', text) is not None

def contains_french(text):
    return (
        re.search(r'[àâæçéèêëîïôœùûüÿ]', text, re.IGNORECASE) is not None or
        re.search(r'\b(le|la|les|un|une|des|ce|cette|est|avec|mais|pour|pas|qui|sur)\b', text, re.IGNORECASE) is not None
    )

def contains_spanish(text):
    return (
        re.search(r'[áéíóúüñ]', text, re.IGNORECASE) is not None or
        re.search(r'\b(el|la|los|las|un|una|que|es|con|pero|por|para|cómo|sin|más)\b', text, re.IGNORECASE) is not None
    )

def contains_italian(text):
    return (
        re.search(r'[àèéìíîòóùú]', text, re.IGNORECASE) is not None or
        re.search(r'\b(il|lo|la|gli|le|un|una|che|è|con|ma|come|perché|senza|più|meno)\b', text, re.IGNORECASE) is not None
    )

def contains_portuguese(text):
    return (
        re.search(r'[áàâãéêíóôõúç]', text, re.IGNORECASE) is not None or
        re.search(r'\b(o|a|os|as|um|uma|que|é|com|mas|por|para|como|sem|mais)\b', text, re.IGNORECASE) is not None
    )

def contains_german(text):
    return (
        re.search(r'[äöüß]', text, re.IGNORECASE) is not None or
        re.search(r'\b(der|die|das|ein|eine|ist|mit|aber|und|nicht|für|ohne|warum|wie|mehr)\b', text, re.IGNORECASE) is not None
    )

def contains_english(text):
    return (
        re.search(r'\b(the|and|is|of|to|in|with|but|not|a|an|for|on|that|how|without|more)\b', text, re.IGNORECASE) is not None
    )

def process_text_block(block_id, text, default_nlp):
    lang_code = detectis_exception_language(text)
    nlp = default_nlp if not lang_code else load_spacy_model(lang_code)
    detected_language = lang_code or "default"
    
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
            structured[s_key]["words"][w_key] = {  # Keep `{` on the same line
               "text": token.text,
               "pos": token.pos_,
               "language": detected_language,
               "ent": token.ent_type_ or None,
               "pinyin": (
                  " ".join(lazy_pinyin(token.text)) 
                  if contains_chinese(token.text) 
                  else None
               )
            }

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
                replacement_content = " ".join([token[0] for token in sentence_tokens])
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


    reformatted_flattened = {}
    for block_id, block_data in structured_output.items():
        # Determine the block type (tag/attr/meta/jsonld)
        block_type = (
            block_data.get("tag") or 
            block_data.get("attr") or 
            block_data.get("meta") or 
            block_data.get("jsonld") or 
            "unknown"
        )
    
        # Get full text (fallback: join all sentences)
        full_text = block_data.get("text", " ".join(
            s_data["text"] for s_data in block_data["tokens"].values()
        ))
    
        reformatted_flattened[block_id] = {
            "type": block_type,  # "p", "alt", "og:title", etc.
            "text": full_text,
            "segments": {  # Renamed from "tokens" for clarity
                f"{block_id}_{s_key}": s_data["text"]
                for s_key, s_data in block_data["tokens"].items()
            }
        }

    with open("translatable_flat.json", "w", encoding="utf-8") as f:
         json.dump(reformatted_flattened, f, indent=2, ensure_ascii=False)
    
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
    # Define supported languages for help text
    SUPPORTED_LANGS = ", ".join(sorted(SPACY_MODELS.keys()))

    parser = argparse.ArgumentParser(
        description="Extract translatable text from HTML.",
        formatter_class=argparse.RawTextHelpFormatter  # For multi-line help
    )
    
    # Required arguments
    parser.add_argument(
        "input_file",
        help="Path to the HTML file to process"
    )
    
    # Primary language (MANDATORY)
    parser.add_argument(
        "--lang",
        choices=SPACY_MODELS.keys(),
        required=True,
        metavar="LANG_CODE",
        help=f"""\
Primary language of the document (REQUIRED).
Supported codes: {SUPPORTED_LANGS}
Examples: --lang en (English), --lang zh (Chinese)"""
    )

    # Secondary language (OPTIONAL)
    parser.add_argument(
        "--secondary-lang",
        choices=SPACY_MODELS.keys(),
        metavar="LANG_CODE",
        help=f"""\
Optional secondary language for mixed-content detection.
Supported codes: {SUPPORTED_LANGS}
Examples: --secondary-lang fr (French), --secondary-lang es (Spanish)"""
    )

    args = parser.parse_args()

    # Validate language priority
    if args.secondary_lang and args.secondary_lang == args.lang:
        parser.error("Primary and secondary languages cannot be the same!")

    # Run extraction
    extract_translatable_html(args.input_file, args.lang)
