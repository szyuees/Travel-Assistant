import bz2, json, re, argparse
from pathlib import Path
from lxml import etree
import mwparserfromhell as mwp
from unidecode import unidecode

def clean_wikicode(text: str) -> str:
    code = mwp.parse(text or "")
    for t in list(code.filter_templates()):
        code.remove(t)
    for l in code.filter_wikilinks():
        l.text = l.text or l.title
    plain = code.strip_code(normalize=True, collapse=True)
    plain = re.sub(r"\s+\[\d+\]", "", plain)
    return unidecode(plain).strip()

def iterate_pages(bz2_path: Path):
    ns = "{http://www.mediawiki.org/xml/export-0.10/}"
    with bz2.open(bz2_path, "rb") as f:
        for _, page in etree.iterparse(f, events=("end",), tag=f"{ns}page"):
            title = page.findtext(f"{ns}title") or ""
            rev = page.find(f"{ns}revision")
            text = rev.findtext(f"{ns}text") if rev is not None else ""
            yield title, text
            page.clear()

def _window(title, url, section, text, win, stride, min_chars):
    i, n = 0, len(text)
    while i < n:
        seg = text[i:i+win]
        if len(seg) >= min_chars:
            yield {
                "id": f"wikivoyage:{title.replace(' ','_')}#{i}",
                "title": title, "section": section, "url": url, "text": seg,
                "license":"CC BY-SA 4.0", "source":"Wikivoyage", "lang":"en"
            }
        i += stride

def chunk_sections(title, url, content, win=1100, stride=900, min_chars=220):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseurl", default="https://en.wikivoyage.org/wiki/")
    ap.add_argument("--win", type=int, default=1100)
    ap.add_argument("--stride", type=int, default=900)
    ap.add_argument("--min-chars", type=int, default=220)
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fout:
        for title, wikitext in iterate_pages(Path(args.dump)):
            if not title or ":" in title:  # 跳过模板/文件命名空间
                continue
            plain = clean_wikicode(wikitext)
            if len(plain) < args.min_chars:
                continue
            url = f"{args.baseurl}{title.replace(' ','_')}"
            for ck in chunk_sections(title, url, plain, args.win, args.stride, args.min_chars):
                fout.write(json.dumps(ck, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
