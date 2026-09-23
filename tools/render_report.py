"""Render the edited Markdown report to PDF without modifying experiment files.

Requires reportlab and matplotlib in addition to the experiment dependencies.
Run from the repository root: python tools/render_report.py
"""
from pathlib import Path
import re
from html import escape
from urllib.parse import urljoin

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether,
    Image, Table, TableStyle, Preformatted, CondPageBreak,
)

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, filename in [
    ("DV", "DejaVuSans.ttf"), ("DV-Bold", "DejaVuSans-Bold.ttf"),
    ("DV-Oblique", "DejaVuSans-Oblique.ttf"), ("DV-Mono", "DejaVuSansMono.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-Bold", italic="DV-Oblique")

INK = colors.HexColor("#18304d")
BLUE = colors.HexColor("#205788")
MUTED = colors.HexColor("#52647a")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("BodyVI", fontName="DV", fontSize=9.3, leading=14.2,
                          textColor=INK, spaceAfter=7, allowWidows=0, allowOrphans=0))
styles.add(ParagraphStyle("TitleVI", parent=styles["BodyVI"], fontName="DV-Bold",
                          fontSize=24, leading=30, spaceAfter=15))
styles.add(ParagraphStyle("PartVI", parent=styles["BodyVI"], fontName="DV-Bold",
                          fontSize=16, leading=22, textColor=BLUE, spaceAfter=12,
                          keepWithNext=True))
styles.add(ParagraphStyle("SectionVI", parent=styles["BodyVI"], fontName="DV-Bold",
                          fontSize=11.3, leading=16, textColor=BLUE,
                          spaceBefore=12, spaceAfter=7, keepWithNext=True))
styles.add(ParagraphStyle("CellVI", parent=styles["BodyVI"], fontSize=8.0, leading=11.8,
                          spaceAfter=0))
styles.add(ParagraphStyle("HeadCellVI", parent=styles["CellVI"], fontName="DV-Bold"))
styles.add(ParagraphStyle("CaptionVI", parent=styles["BodyVI"], fontSize=8.4,
                          leading=12, alignment=TA_CENTER, textColor=MUTED,
                          spaceBefore=5, spaceAfter=12))
styles.add(ParagraphStyle("CodeVI", parent=styles["BodyVI"], fontName="DV-Mono",
                          fontSize=7.0, leading=10.5, backColor=colors.HexColor("#f0f4f8"),
                          borderPadding=8, spaceBefore=4, spaceAfter=11))

REPO_FILES = "https://github.com/cau26/CO3133-DLProject/blob/main/"

def inline(text):
    # Keep generated XML safe; resolve relative document links for the portable PDF.
    tokens = {}
    def link(m):
        key = f"LINKTOKEN{len(tokens)}END"
        target = m.group(2)
        if not re.match(r"^https?://", target):
            target = urljoin(REPO_FILES, target)
        tokens[key] = f'<a href="{escape(target, quote=True)}" color="#205788">{escape(m.group(1))}</a>'
        return key
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    for key, value in tokens.items():
        text = text.replace(key, value)
    return text

def footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(colors.HexColor("#d5e0eb"))
    canvas.setLineWidth(0.5)
    canvas.line(18*mm, 17*mm, w-18*mm, 17*mm)
    canvas.setFont("DV", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(18*mm, 12*mm, "CO3133 | G-M10 | Assignment 1 - M1 Draft")
    canvas.drawRightString(w-18*mm, 12*mm, str(doc.page))
    canvas.restoreState()

def render():
    source = ROOT / "assignment1.md"
    target = ROOT / "G-M10_A1_Draft.pdf"
    width = A4[0] - 36*mm
    doc = SimpleDocTemplate(
        str(target), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
        topMargin=17*mm, bottomMargin=23*mm, title="G-M10 - Assignment 1 M1 Draft",
        author="G-M10; edited with disclosed AI assistance",
    )
    lines = source.read_text().splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "---":
            i += 1
            continue
        if line.startswith("~~~"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("~~~"):
                block.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(block), styles["CodeVI"], maxLineLength=100))
            i += 1
            continue
        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image_match:
            asset = ROOT / image_match.group(2)
            image = Image(str(asset))
            scale = min(width/image.imageWidth, 350/image.imageHeight)
            image.drawWidth = image.imageWidth*scale
            image.drawHeight = image.imageHeight*scale
            image.hAlign = "CENTER"
            group = [Spacer(1, 6), image,
                     Paragraph(inline(image_match.group(1)), styles["CaptionVI"])]
            while story and isinstance(story[-1], Paragraph) and story[-1].style.name in ("SectionVI", "PartVI"):
                group.insert(0, story.pop())
            story.append(KeepTogether(group))
            i += 1
            continue
        if line.startswith("|"):
            raw = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                parts = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in parts):
                    raw.append(parts)
                i += 1
            n = len(raw[0])
            assert all(len(row) == n for row in raw), "Inconsistent table width"
            if n == 2:
                fractions = [0.34, 0.66]
            elif n == 3:
                fractions = [0.24, 0.15, 0.61] if raw[0][0] == "Thành viên" else [0.14, 0.68, 0.18]
            elif n == 4:
                fractions = [0.31, 0.23, 0.17, 0.29]
            elif n == 5:
                fractions = [0.29, 0.16, 0.16, 0.195, 0.195]
            elif raw[0][1] == "Train (s)":
                fractions = [0.13, 0.13, 0.17, 0.15, 0.21, 0.21]
            else:
                fractions = [0.15, 0.14, 0.18, 0.20, 0.18, 0.15]
            cells = [[Paragraph(inline(c), styles["HeadCellVI"] if j == 0 else styles["CellVI"])
                      for c in row] for j, row in enumerate(raw)]
            table = Table(cells, colWidths=[width*f for f in fractions], repeatRows=1,
                          hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cfdae6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            group = [Spacer(1, 3), table, Spacer(1, 10)]
            while story and isinstance(story[-1], Paragraph) and story[-1].style.name in ("SectionVI", "PartVI"):
                group.insert(0, story.pop())
            story.append(KeepTogether(group))
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["TitleVI"]))
            i += 1
            continue
        if line.startswith("## "):
            story.append(PageBreak() if line.startswith("## Phần 1.") else CondPageBreak(170))
            story.append(Paragraph(inline(line[3:]), styles["PartVI"]))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["SectionVI"]))
            i += 1
            continue
        if re.match(r"^\d+\.\s", line):
            story.append(Paragraph(inline(line), styles["BodyVI"]))
            i += 1
            continue
        paragraph = []
        while i < len(lines):
            p = lines[i].strip()
            if not p or p.startswith(("#", "|", "~~~", "![")) or re.match(r"^\d+\.\s", p):
                break
            paragraph.append(inline(p))
            if lines[i].endswith("  "):
                paragraph.append("<br/>")
            i += 1
        if paragraph:
            story.append(Paragraph(" ".join(paragraph), styles["BodyVI"]))
        else:
            i += 1
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(target)

if __name__ == "__main__":
    render()
