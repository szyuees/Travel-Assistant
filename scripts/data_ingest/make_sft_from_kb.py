# scripts/data_ingest/make_sft_from_kb.py
import json, random, re, argparse, pathlib, collections

SPLIT = re.compile(r"(?<=[.!?])\s+")
# 放宽：所有这些都算“旅行相关”小节；如果为空也允许
SECTIONS_OK = {
    "see","do","get around","transport","understand","eat","drink","sleep",
    "get in","buy","stay","learn","work","connect","go next","intro","introduction"
}

PROMPT = (
    "Answer ONLY with bullet points using the EVIDENCE below.\n"
    "Question: {q}\n\n"
    "EVIDENCE (title — URL, then a sentence):\n{ctx}\n\n"
    "Bulleted answer (2–4 bullets, each ends with (Title — URL)):\n"
)

def pick_sents(text, k=3, min_len=20, max_len=320):
    sents = [t.strip() for t in SPLIT.split(text or "") if min_len <= len(t) <= max_len]
    random.shuffle(sents)
    return sents[:k]

def build_question(title, sec):
    sec = (sec or "").lower()
    if "transport" in sec or "get around" in sec:
        return f"What is the best way to get around in {title}?"
    if "see" in sec:
        return f"What should I visit in {title}?"
    if "do" in sec:
        return f"What are fun things to do in {title}?"
    return f"What should I know when visiting {title}?"

def exs_from_doc(d):
    title = d.get("title","").strip() or d.get("source","").strip()
    url   = d.get("url","").strip()
    sec   = (d.get("section") or "").lower()

    # 放宽：未知章节也允许，只要有文本
    if sec and sec not in SECTIONS_OK:
        return []

    text = d.get("text") or d.get("content") or d.get("body") or ""
    sents = pick_sents(text, k=3)
    if not (title and url and sents):
        return []

    q = build_question(title, sec)
    ctx = "\n".join([f"[{title}] {url}\n{t}" for t in sents])
    bullets = [f"• {t} ({title} — {url})" for t in sents[:3]]
    return [{"input": PROMPT.format(q=q, ctx=ctx), "target": "\n".join(bullets), "title": title, "section": sec, "url": url}]

def main(kb, out_train, out_val, max_docs=None, seed=42, show_samples=3):
    random.seed(seed)
    exs = []
    sec_counter = collections.Counter()
    kept_counter = collections.Counter()

    with open(kb, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
            except:
                continue
            sec = (d.get("section") or "").lower()
            sec_counter[sec] += 1
            cur = exs_from_doc(d)
            if cur:
                kept_counter[sec] += 1
                exs += cur
            if max_docs and i+1 >= max_docs:
                break

    # 打印统计
    print("Total examples kept:", len(exs))
    print("Top sections in KB:", sec_counter.most_common(15))
    print("Kept by section:", kept_counter.most_common(15))

    random.shuffle(exs)
    n = max(1, int(len(exs) * 0.9)) if exs else 0
    pathlib.Path(out_train).write_text(json.dumps(exs[:n], ensure_ascii=False, indent=2))
    pathlib.Path(out_val).write_text(json.dumps(exs[n:], ensure_ascii=False, indent=2))
    print(f"wrote {len(exs[:n])} train / {len(exs[n:])} val")

    # 打印几条样例，便于人工检查
    for i, e in enumerate(exs[:show_samples]):
        print(f"\n=== SAMPLE {i+1} ===")
        print("INPUT:\n", e["input"][:600])
        print("TARGET:\n", e["target"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--train_out", default="data/train_supervised.json")
    ap.add_argument("--val_out", default="data/val_supervised.json")
    ap.add_argument("--max_docs", type=int, default=None)  # 不再默认截断
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.kb, args.train_out, args.val_out, args.max_docs, args.seed)
