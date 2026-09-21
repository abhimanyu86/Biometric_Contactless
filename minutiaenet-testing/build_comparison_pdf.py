"""
Renders COMPARISON.md into COMPARISON.pdf. Parses the actual markdown file
(headings, tables, bullet lists, paragraphs) rather than duplicating its
content by hand, so the PDF can't drift from the source doc.
"""
import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE_DIR, "COMPARISON.md")
OUT = os.path.join(BASE_DIR, "COMPARISON.pdf")

NAVY = colors.HexColor("#0B2545")
ACCENT = colors.HexColor("#134074")
LIGHT = colors.HexColor("#EEF2F7")
GREY = colors.HexColor("#444444")

styles = getSampleStyleSheet()


def S(name, **kw):
    base = kw.pop("parent", styles["Normal"])
    return ParagraphStyle(name, parent=base, **kw)


title_style = S("TitleX", fontName="Helvetica-Bold", fontSize=18, textColor=NAVY, leading=22, spaceAfter=10)
h2_style = S("H2X", fontName="Helvetica-Bold", fontSize=13, textColor=ACCENT, leading=16, spaceBefore=14, spaceAfter=6)
h3_style = S("H3X", fontName="Helvetica-Bold", fontSize=11, textColor=ACCENT, leading=14, spaceBefore=10, spaceAfter=4)
body_style = S("BodyX", fontName="Helvetica", fontSize=9.5, textColor=GREY, leading=13.5, alignment=TA_LEFT, spaceAfter=6)
bullet_style = S("BulletX", parent=body_style, spaceAfter=2)
code_style = S("CodeX", fontName="Courier", fontSize=8.5, textColor=NAVY, leading=12)


def inline_md_to_html(text):
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" color="#0B2545">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)  # drop links, keep label
    return text


def parse_markdown(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        if line.startswith("# "):
            blocks.append(("h1", line[2:].strip()))
            i += 1
        elif line.startswith("## "):
            blocks.append(("h2", line[3:].strip()))
            i += 1
        elif line.startswith("### "):
            blocks.append(("h3", line[4:].strip()))
            i += 1
        elif line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = [
                [cell.strip() for cell in row.strip().strip("|").split("|")]
                for row in table_lines
                if not re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?$", row)
            ]
            blocks.append(("table", rows))
        elif line.strip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:])
                i += 1
            blocks.append(("list", items))
        else:
            para_lines = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|")) and not lines[i].strip().startswith("- "):
                para_lines.append(lines[i])
                i += 1
            blocks.append(("p", " ".join(para_lines)))

    return blocks


def build_pdf():
    blocks = parse_markdown(SRC)
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        topMargin=18 * mm, bottomMargin=16 * mm, leftMargin=16 * mm, rightMargin=16 * mm,
        title="MinutiaeNet matching test: with vs without preprocessing",
    )
    story = []

    for kind, content in blocks:
        if kind == "h1":
            story.append(Paragraph(inline_md_to_html(content), title_style))
        elif kind == "h2":
            story.append(Paragraph(inline_md_to_html(content), h2_style))
        elif kind == "h3":
            story.append(Paragraph(inline_md_to_html(content), h3_style))
        elif kind == "p":
            story.append(Paragraph(inline_md_to_html(content), body_style))
        elif kind == "list":
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(inline_md_to_html(item), bullet_style)) for item in content],
                    bulletType="bullet",
                    leftIndent=12,
                )
            )
            story.append(Spacer(1, 4))
        elif kind == "table":
            data = [[Paragraph(inline_md_to_html(cell), body_style) for cell in row] for row in content]
            table = Table(data, hAlign="LEFT", repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 8))

    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build_pdf()
