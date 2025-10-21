#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bz2, json, re, argparse
from pathlib import Path
from lxml import etree
import mwparserfromhell as mwp
from unidecode import unidecode
import mwparserfromhell

def clean_wikicode(text: str) -> str:
    code = mwp.parse(text or "")
    plain = code.strip_code(normalize=True, collapse=True)
    plain = re.sub(r"\s*\[\d+\]\s*", " ", plain)           # 去脚注编号
    plain = re.sub(r"(?mi)^(thumb\|.*)$", "", plain)        # 去图片图注行
    plain = re.sub(r"(?mi)\b(File|Image):\S+\b", "", plain) # 去 File:/Image: 残留
    plain = re.sub(r"[ \t]+\n", "\n", plain)                # 去尾随空白
    return unidecode(plain).strip()



def iterate_pages(bz2_path: Path, max_pages: int | None = None):
    """
    兼容任意 export-*.*/ 命名空间的 MediaWiki dump 解析：
    动态识别 {namespace}，避免固定 export-0.10/0.11 导致 0 匹配。
    """
    seen = 0
    with bz2.open(bz2_path, "rb") as f:
        # 不预设 tag，逐个元素检查
        context = etree.iterparse(f, events=("end",))
        for _, elem in context:
            tag = elem.tag  # e.g. "{http://www.mediawiki.org/xml/export-0.11/}page"
            if isinstance(tag, str) and tag.endswith("}page"):
                ns = tag.split("}")[0] + "}"  # "{...}"
                title = elem.findtext(f"{ns}title") or ""
                rev = elem.find(f"{ns}revision")
                text = rev.findtext(f"{ns}text") if rev is not None else ""
                yield title, text
                elem.clear()
                seen += 1
                if max_pages and seen >= max_pages:
                    break

def _window(title, url, section, text, win, stride, min_chars):
    i, n = 0, len(text)
    while i < n:
        seg = text[i:i+win]
        if len(seg) >= min_chars:
            yield {
                "id": f"wikivoyage:{title.replace(' ','_')}#{i}",
                "title": title,
                "section": section,
                "url": url,
                "text": seg,
                "license": "CC BY-SA 4.0",
                "source": "Wikivoyage",
                "lang": "en",
            }
        i += stride

def chunk_sections(title, url, content, win=1100, stride=900, min_chars=220):
    # 先按二/三级标题切段，再滑窗
    parts = re.split(r"(\n==+ .+? ==+\n)", content)
    section, buf = "intro", []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            m = re.search(r"==+\s(.+?)\s==+", part)
            if m:
                if buf:
                    yield from _window(title, url, section, "".join(buf).strip(), win, stride, min_chars)
                section, buf = m.group(1), []
        else:
            buf.append(part)
    if buf:
        yield from _window(title, url, section, "".join(buf).strip(), win, stride, min_chars)

def main():
    ap = argparse.ArgumentParser("Build chunked KB JSONL from Wikivoyage dump")
    ap.add_argument("--dump", required=True, help="path to enwikivoyage-...pages-articles.xml.bz2")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--baseurl", default="https://en.wikivoyage.org/wiki/")
    ap.add_argument("--win", type=int, default=1100)
    ap.add_argument("--stride", type=int, default=900)
    ap.add_argument("--min-chars", type=int, default=180)  # 放宽一点，先确保有产出
    ap.add_argument("--max-pages", type=int, default=None, help="for quick test; e.g., 200")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    with open(out, "w", encoding="utf-8") as fout:
        for title, wikitext in iterate_pages(Path(args.dump), max_pages=args.max_pages):
            # 跳过模板/文件等命名空间页
            if not title or ":" in title:
                continue
            plain = clean_wikicode(wikitext)
            if len(plain) < args.min_chars:
                continue
            url = f"{args.baseurl}{title.replace(' ', '_')}"
            for ck in chunk_sections(title, url, plain, args.win, args.stride, args.min_chars):
                fout.write(json.dumps(ck, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"[done] wrote {total_chunks} chunks -> {out}")

if __name__ == "__main__":
    main()
