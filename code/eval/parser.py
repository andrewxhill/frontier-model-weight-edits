"""Turn a model's reply into one option from the list it was shown.

Accept an option letter, a name that matches exactly one option, or a letter and
name that agree. Ambiguous names and conflicting pairs stay unresolved. The raw
reply is never changed, and refusals, truncated, looping or empty replies stay faults.
"""
import json
import re


def normalized_name(value):
    return ' '.join(value.strip().rstrip('.!').split()).casefold()


def parse_answer(text, options, fault=None):
    """Return a canonical label or None, without guessing from extra prose."""
    if fault not in (None, 'malformed') or not isinstance(text, str):
        return None
    delivered = text.rsplit('</think>', 1)[-1] if '</think>' in text else text
    if '<think>' in delivered:
        return None
    delivered = delivered.strip().strip('`').strip()
    try:
        value = json.loads(delivered)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, dict):
        delivered = value.get('answer')
        if not isinstance(delivered, str):
            return None
        delivered = delivered.strip()
    elif value is not None:
        return None
    delivered = re.sub(r'^answer\s*[:=]\s*', '', delivered, count=1, flags=re.I).strip()
    by_letter = {option['letter'].upper(): option for option in options}
    matches = [option['label'] for option in options
               if normalized_name(delivered) == normalized_name(option['name'])]
    if matches:
        return matches[0] if len(matches) == 1 else None
    pair = re.fullmatch(r'([A-Za-z]+)\s*[.)\]:-]\s*(.+)', delivered)
    if pair:
        option = by_letter.get(pair[1].upper())
        return (option['label'] if option is not None and
                normalized_name(pair[2]) == normalized_name(option['name']) else None)
    candidate = delivered.rstrip('.!').strip().upper()
    if candidate in by_letter:
        return by_letter[candidate]['label']
    return None

def _legacy_parse_answer(text, choices):
    text = text.rsplit('</think>', 1)[-1] if '</think>' in text else text
    if '<think>' in text:  # Unclosed reasoning isn't a delivered answer.
        return None
    text = text.strip().strip('`').strip()
    try:
        value = json.loads(text)
        text = str(value['answer']) if isinstance(value, dict) else text
    except (ValueError, KeyError):
        match = re.fullmatch(r'(?:answer\s*[:=]\s*)?[\s"\']*([A-Z]|[a-z_-]+)[\s"\'.!]*', text, re.I)
        text = match[1] if match else text
    letters = {chr(65+i): label for i, label in enumerate(choices)}
    if text.upper() in letters and len(text) == 1:
        return letters[text.upper()]
    return next((v for v in choices if text.casefold() == v.casefold()), None)


def is_loop(tokens):
    # At least 32 repeated tokens, four repetitions, in the terminal 512-token window.
    tail = list(tokens[-512:])
    for width in range(1, min(128, len(tail)//4)+1):
        unit = tail[-width:]
        repeats = 0
        for end in range(len(tail), width-1, -width):
            if tail[end-width:end] != unit:
                break
            repeats += 1
        if repeats >= 4 and repeats*width >= 32:
            return True
    return False


def classify_reply(text, tokens, choices, *, max_tokens, infrastructure_error=None):
    if infrastructure_error:
        return 'infrastructure'
    if is_loop(tokens):
        return 'loop'
    if len(tokens) >= max_tokens:
        return 'truncation'
    if not text.strip():
        return 'empty'
    if re.search(r"\b(cannot|can't|unable to|sorry|refuse)\b", text, re.I) and _legacy_parse_answer(text, choices) is None:
        return 'refusal'
    return None if _legacy_parse_answer(text, choices) is not None else 'malformed'



def delivered_answer(raw, tokens, options, max_tokens):
    fault = classify_reply(raw, tokens, [o['name'] for o in options], max_tokens=max_tokens)
    answer = parse_answer(raw, options, fault=fault)
    if answer is None and fault is None:
        fault = 'malformed'
    if answer is not None and fault == 'malformed':
        fault = None
    return answer, fault
