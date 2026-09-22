"""Export Markdown to a readable, printable HTML file with embedded local images.

Example: python export_html.py outputs/a1/draft_report.md
Open the resulting .html in a browser. Print -> Save as PDF if needed.
"""
import argparse
import base64
import html
import mimetypes
import re
from pathlib import Path
from urllib.parse import unquote

import markdown

CSS = """
:root {color-scheme:light} * {box-sizing:border-box}
body {font:16px/1.65 system-ui,Segoe UI,Arial,sans-serif;color:#172033;
      background:#f4f6fa;margin:0;padding:32px 20px}
main {max-width:1060px;margin:auto;background:white;padding:42px 48px;border-radius:12px}
h1 {font-size:30px;line-height:1.25;color:#173c77} h2 {margin-top:38px;font-size:23px;color:#173c77}
h3 {margin-top:28px;font-size:19px} a {color:#1557b0;overflow-wrap:anywhere}
p,li {overflow-wrap:anywhere} table {border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}
th,td {border:1px solid #d6deeb;text-align:left;padding:9px 11px;vertical-align:top;overflow-wrap:anywhere}
th {background:#edf3fd} tr:nth-child(even) {background:#fafbfd}
pre {background:#edf2f8;padding:15px;border-radius:6px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}
code {font-family:Consolas,ui-monospace,monospace} :not(pre)>code {background:#edf2f8;padding:1px 4px}
img {display:block;max-width:100%;height:auto;margin:20px auto}
blockquote {border-left:4px solid #3474c5;background:#f0f6ff;margin:20px 0;padding:8px 18px}
@media(max-width:650px) {body{padding:8px}main{padding:20px 16px}table{font-size:12px}th,td{padding:6px}}
@media print {@page{size:A4;margin:16mm}body{background:white;padding:0;font-size:10pt}
 main{max-width:none;padding:0;border-radius:0}h1{font-size:22pt}h2{font-size:16pt}
 h1,h2,h3{break-after:avoid}img{max-height:235mm;object-fit:contain;break-inside:avoid}
 tr,pre,blockquote{break-inside:avoid}a{color:inherit}table{font-size:8.5pt}}
"""


def export(source, destination=None):
    source = Path(source).resolve()
    destination = Path(destination) if destination else source.with_suffix(".html")
    text = source.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "toc"])

    def embed(match):
        url = html.unescape(match.group(2))
        if url.startswith(("https://", "http://", "data:")):
            return match.group(0)
        path = (source.parent / unquote(url)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'{match.group(1)}data:{mime};base64,{data}{match.group(3)}'

    body = re.sub(r'(<img\b[^>]*\bsrc=")([^"]+)(")', embed, body)
    title = next((line.lstrip("# ") for line in text.splitlines() if line.startswith("# ")), source.stem)
    document = ('<!doctype html><html lang="vi"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>'
                + body + '</main></body></html>')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    print(f"HTML ready: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output")
    args = parser.parse_args()
    export(args.source, args.output)
