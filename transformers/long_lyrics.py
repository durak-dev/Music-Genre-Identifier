import re
import numpy as np

MAX_TOKENS = 510                             # 512 minus the two special tokens (<s>, </s>)
HEAD_TOKENS = 128                            # head + tail split  (2019): 128 + 382
TAIL_TOKENS = MAX_TOKENS - HEAD_TOKENS - 1   # one token is used by the line break joining head and tail


def token_counts(texts, tokenizer, batch_size=1000):
    counts = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], add_special_tokens=False)["input_ids"]
        counts.extend(len(ids) for ids in batch)
    return np.array(counts)


def _line_key(line):
    # Lines count as repeats regardless of casing and surrounding whitespace
    return line.strip().lower()


def repeated_lines(text):
    """Returns (non-empty lines, lines that repeat an earlier line of the same song)."""
    seen, n_lines, n_repeated = set(), 0, 0
    for line in text.split("\n"):
        key = _line_key(line)
        if not key:
            continue
        n_lines += 1
        if key in seen:
            n_repeated += 1
        seen.add(key)
    return n_lines, n_repeated


def remove_repeated_lines(text):
    """Keeps only the first occurrence of every line (e.g. the chorus is kept once)."""
    seen, kept = set(), []
    for line in text.split("\n"):
        key = _line_key(line)
        if key in seen:
            continue
        if key:
            seen.add(key)
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def head_tail(text, tokenizer, head=HEAD_TOKENS, tail=TAIL_TOKENS):
    """Keeps the first `head` and the last `tail` tokens of a text that is too long, joined by a line break."""
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    limit = head + tail + 1
    if len(offsets) <= limit:
        return text
    # Tokens can merge differently at the new seam, so the result is re-counted and the tail shortened if needed
    while True:
        truncated = text[:offsets[head - 1][1]] + "\n" + text[offsets[-tail][0]:]
        excess = len(tokenizer(truncated, add_special_tokens=False)["input_ids"]) - limit
        if excess <= 0:
            return truncated
        tail -= excess


def split_in_two(text, tokenizer, max_tokens=MAX_TOKENS):
    """Splits a text that is too long into two halves of roughly equal token length.

    The cut is placed at the stanza break (blank line) closest to the middle token, as long as one lies
    in the middle third of the song; otherwise at the closest line break.
    """
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    n = len(offsets)
    if n <= max_tokens:
        return [text]

    middle = offsets[n // 2][0]
    low, high = offsets[n // 3][0], offsets[2 * n // 3][0]
    stanza_breaks = [m.start() for m in re.finditer(r"\n\n", text) if low <= m.start() <= high]
    breaks = stanza_breaks or [m.start() for m in re.finditer(r"\n", text)]
    cut = min(breaks, key=lambda b: abs(b - middle)) if breaks else middle
    return [text[:cut].strip(), text[cut:].strip()]


def average_part_scores(part_scores, part_lengths):
    """Song-level scores as the mean of its parts' scores, weighted by the parts' token lengths."""
    return np.average(np.asarray(part_scores), axis=0, weights=part_lengths)
