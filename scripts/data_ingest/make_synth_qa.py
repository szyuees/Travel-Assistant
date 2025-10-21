import json, re, argparse
from random import sample

TEMPLATES = [
  ("How do I get from the airport to the city center in {title}?",
   "From the guide: {snippet} (Source: {title} → {section})"),
  ("What public transport options are available in {title}?",
   "According to the guide: {snippet} (Source: {title} → {section})"),
]

def pick_snippet(txt):
    sents = re.split(r"(?<=[.!?])\s+", txt)
    s = " ".join(sents[:3]) if sents else txt[:300]
    return s[:400]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-chunk", type=int, default=1)
    args = ap.parse_args()

    with open(args.chunks, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            ck = json.loads(line)
            text = ck.get("text","")
            if len(text) < 300:
                continue
            snippet = pick_snippet(text)
            for qfmt, afmt in sample(TEMPLATES, k=min(args.per_chunk, len(TEMPLATES))):
                q = qfmt.format(title=ck.get("title",""))
                a = afmt.format(snippet=snippet, title=ck.get("title",""), section=ck.get("section",""))
                fout.write(json.dumps({"question": q, "answer": a, "doc_id": ck.get("id"), "url": ck.get("url")}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
