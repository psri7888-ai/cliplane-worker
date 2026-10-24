"""Highlight selection: an LLM proposes moments, deterministic code snaps them to
sentence boundaries and scores them from listed signals. No fallback results are invented."""
import json, os, re, requests

SCORE_WEIGHTS = {"hook": 25, "pacing": 25, "emphasis": 15, "clean_ending": 20, "duration_fit": 15}


def propose(segments, n, min_s, max_s):
    key, gkey = os.environ.get("ANTHROPIC_API_KEY"), os.environ.get("GROQ_API_KEY")
    if not key and not gkey:
        raise RuntimeError("GROQ_API_KEY (free) or ANTHROPIC_API_KEY is not set. Highlight detection needs an AI provider key.")
    text = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments)[:150000]
    prompt = (f"Pick up to {n} self-contained moments from this transcript that would work as standalone short clips, "
              f"each between {min_s} and {max_s} seconds. Start and end on complete sentences. "
              "Reply with ONLY a JSON list of objects: start, end (seconds), title (max 70 chars), "
              "description (one sentence). Do not promise views or virality.\n\n" + text)
    if not key:  # free path: Groq's OpenAI-compatible chat endpoint
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=120,
                          headers={"Authorization": f"Bearer {gkey}"},
                          json={"model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                                "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000})
        if r.status_code != 200:
            raise RuntimeError(f"AI provider returned {r.status_code}. Check GROQ_API_KEY and GROQ_MODEL (see console.groq.com models list).")
        raw = r.json()["choices"][0]["message"]["content"]
        return _parse(raw)
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=120,
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                      json={"model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"), "max_tokens": 2000,
                            "messages": [{"role": "user", "content": prompt}]})
    if r.status_code != 200:
        raise RuntimeError(f"AI provider returned {r.status_code}. Check ANTHROPIC_API_KEY and model name.")
    return _parse("".join(b.get("text", "") for b in r.json()["content"]))


def _parse(raw):
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        raise RuntimeError("AI provider reply did not contain a clip list.")
    return json.loads(m.group(0))


def snap(start, end, segments, max_s):
    """Move start back to the nearest segment start and end forward to a segment end."""
    starts = [s["start"] for s in segments if s["start"] <= start + 0.25]
    s0 = max(starts) if starts else start
    ends = [s["end"] for s in segments if s["end"] >= end - 0.25]
    e0 = min(ends) if ends else end
    if e0 - s0 > max_s:  # too long after snapping: end on the last segment that still fits
        fit = [s["end"] for s in segments if s0 < s["end"] <= s0 + max_s]
        e0 = max(fit) if fit else s0 + max_s
    return s0, e0


def score(start, end, segments, min_s, max_s):
    inside = [s for s in segments if s["end"] > start and s["start"] < end]
    txt = " ".join(s["text"] for s in inside).strip()
    dur = max(end - start, 0.1)
    first = inside[0]["text"] if inside else ""
    wps = len(txt.split()) / dur
    sig = {
        "hook": 1.0 if ("?" in first or re.search(r"\d", first)) else (0.5 if len(first.split()) <= 12 else 0.2),
        "pacing": max(0.0, 1 - abs(wps - 2.7) / 2.7),  # ~2.7 words/s is a lively speaking rate
        "emphasis": min(1.0, (txt.count("!") + txt.count("?")) / 3),
        "clean_ending": 1.0 if txt.endswith((".", "?", "!")) else 0.3,
        "duration_fit": 1.0 if min_s <= dur <= max_s else 0.3,
    }
    total = round(sum(SCORE_WEIGHTS[k] * v for k, v in sig.items()))
    return total, {"weights": SCORE_WEIGHTS, "values": {k: round(v, 2) for k, v in sig.items()},
                   "note": "Guidance from transcript signals only. Not a prediction of views."}


def build_clips(segments, video_duration, n=3, min_s=20, max_s=60):
    out, used = [], []
    for c in propose(segments, n, min_s, max_s):
        try:
            s, e = snap(float(c["start"]), float(c["end"]), segments, max_s)
        except (KeyError, TypeError, ValueError):
            continue
        e = min(e, video_duration)
        if e - s < 5 or any(s < ue and e > us for us, ue in used):
            continue
        used.append((s, e))
        total, sig = score(s, e, segments, min_s, max_s)
        text = " ".join(x["text"] for x in segments if x["end"] > s and x["start"] < e)
        out.append({"title": str(c.get("title", "Untitled clip"))[:150], "description": str(c.get("description", "")),
                    "start_seconds": round(s, 2), "end_seconds": round(e, 2), "score": total, "score_signals": sig})
    return out
