"""Generate the printable DeepSignal project guide using only Python's stdlib."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "DeepSignal_Project_Guide.md"
OUTPUT = ROOT / "DeepSignal_Project_Guide.pdf"

PAGE_W, PAGE_H = 595.28, 841.89
LEFT, RIGHT = 54.0, 54.0
CONTENT_W = PAGE_W - LEFT - RIGHT
TOP, BOTTOM = 68.0, 785.0

NAVY = (0.055, 0.075, 0.14)
INK = (0.12, 0.16, 0.24)
MUTED = (0.38, 0.43, 0.52)
BLUE = (0.17, 0.37, 0.85)
PALE = (0.94, 0.96, 0.99)
LINE = (0.86, 0.89, 0.93)


def pdf_string(value: str) -> str:
    value = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
    value = value.replace("`", "")
    substitutions = {
        "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00b1": "+/-",
        "\u2192": "->", "\u2265": ">=", "\u2264": "<=", "\u2022": "*",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u00a0": " ", "\u2026": "...",
    }
    for old, new in substitutions.items():
        value = value.replace(old, new)
    return value


def hex_pdf_text(value: str) -> str:
    raw = pdf_string(value).encode("cp1252", errors="replace")
    return "<" + raw.hex().upper() + ">"


def color_fill(rgb):
    return "%.3f %.3f %.3f rg" % rgb


def color_stroke(rgb):
    return "%.3f %.3f %.3f RG" % rgb


class Canvas:
    def __init__(self):
        self.commands: list[str] = []
        self.y = TOP
        self.has_content = False

    def text(self, x, top, value, size=10, font="F1", color=INK):
        self.commands.extend([
            color_fill(color), "BT", f"/{font} {size:.2f} Tf",
            f"1 0 0 1 {x:.2f} {PAGE_H-top:.2f} Tm",
            f"{hex_pdf_text(value)} Tj", "ET",
        ])

    def rect(self, x, top, w, h, fill, stroke=None, line_width=0.7):
        self.commands.append(color_fill(fill))
        if stroke:
            self.commands.extend([color_stroke(stroke), f"{line_width:.2f} w"])
            self.commands.append(f"{x:.2f} {PAGE_H-top-h:.2f} {w:.2f} {h:.2f} re B")
        else:
            self.commands.append(f"{x:.2f} {PAGE_H-top-h:.2f} {w:.2f} {h:.2f} re f")

    def line(self, x1, top1, x2, top2, color=LINE, width=0.7):
        self.commands.extend([
            color_stroke(color), f"{width:.2f} w", "[] 0 d",
            f"{x1:.2f} {PAGE_H-top1:.2f} m {x2:.2f} {PAGE_H-top2:.2f} l S",
        ])

    def circle(self, x, top, r, fill):
        # Four cubic curves provide a portable filled circle.
        k = 0.55228475 * r
        cy = PAGE_H - top
        self.commands.extend([
            color_fill(fill),
            f"{x+r:.2f} {cy:.2f} m",
            f"{x+r:.2f} {cy+k:.2f} {x+k:.2f} {cy+r:.2f} {x:.2f} {cy+r:.2f} c",
            f"{x-k:.2f} {cy+r:.2f} {x-r:.2f} {cy+k:.2f} {x-r:.2f} {cy:.2f} c",
            f"{x-r:.2f} {cy-k:.2f} {x-k:.2f} {cy-r:.2f} {x:.2f} {cy-r:.2f} c",
            f"{x+k:.2f} {cy-r:.2f} {x+r:.2f} {cy-k:.2f} {x+r:.2f} {cy:.2f} c f",
        ])


def split_markdown(source: str):
    lines = source.splitlines()
    first_chapter = next(i for i, line in enumerate(lines) if line.startswith("## "))
    title = next(line[2:].strip() for line in lines[:first_chapter] if line.startswith("# "))
    cover_meta = []
    for line in lines[:first_chapter]:
        line = line.strip()
        if line.startswith("> "):
            cover_meta.append(line[2:])
        elif line and not line.startswith("# "):
            cover_meta.append(line)

    blocks = []
    paragraph = []
    code = []
    in_code = False
    for line in lines[first_chapter:]:
        if line.startswith("## "):
            if paragraph:
                blocks.append(("p", " ".join(paragraph)))
                paragraph = []
            blocks.append(("chapter", line[3:].strip()))
            continue
        if line.startswith("### "):
            if paragraph:
                blocks.append(("p", " ".join(paragraph)))
                paragraph = []
            blocks.append(("sub", line[4:].strip()))
            continue
        if line.strip().startswith("```"):
            if in_code:
                blocks.append(("code", code[:]))
                code = []
                in_code = False
            else:
                if paragraph:
                    blocks.append(("p", " ".join(paragraph)))
                    paragraph = []
                in_code = True
            continue
        if in_code:
            code.append(line.rstrip())
            continue
        if not line.strip():
            if paragraph:
                blocks.append(("p", " ".join(paragraph)))
                paragraph = []
            continue
        if line.startswith("- "):
            if paragraph:
                blocks.append(("p", " ".join(paragraph)))
                paragraph = []
            blocks.append(("bullet", line[2:].strip()))
        elif line.startswith("> "):
            if paragraph:
                blocks.append(("p", " ".join(paragraph)))
                paragraph = []
            blocks.append(("callout", line[2:].strip()))
        else:
            paragraph.append(line.strip())
    if paragraph:
        blocks.append(("p", " ".join(paragraph)))
    if in_code:
        blocks.append(("code", code))
    chapters = [text for kind, text in blocks if kind == "chapter"]
    return title, cover_meta, chapters, blocks


def wrap(text: str, max_chars: int):
    text = pdf_string(text)
    return textwrap.wrap(
        text,
        width=max(12, max_chars),
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=True,
    ) or [""]


def add_page(pages):
    canvas = Canvas()
    canvas.rect(0, 0, PAGE_W, PAGE_H, (1, 1, 1))
    canvas.text(LEFT, 38, "DEEPSIGNAL  /  PROJECT GUIDE", 8, "F2", MUTED)
    canvas.text(PAGE_W-RIGHT-150, 38, "IMPLEMENTATION DOCUMENTATION", 7.2, "F1", MUTED)
    canvas.line(LEFT, 48, PAGE_W-RIGHT, 48, LINE, 0.6)
    canvas.line(LEFT, 800, PAGE_W-RIGHT, 800, LINE, 0.6)
    pages.append(canvas)
    canvas.y = TOP
    return canvas


def ensure_space(canvas, pages, height):
    if canvas.y + height > BOTTOM:
        canvas = add_page(pages)
    return canvas


def draw_cover(title, meta):
    c = Canvas()
    c.rect(0, 0, PAGE_W, PAGE_H, NAVY)
    c.rect(0, 0, 13, PAGE_H, BLUE)
    c.rect(54, 102, 54, 5, (0.39, 0.55, 1.0))
    c.text(54, 140, "PROJECT GUIDE  /  FULL TECHNICAL WALKTHROUGH", 9, "F2", (0.70, 0.77, 0.93))
    y = 236
    for line in wrap(title, 27):
        c.text(54, y, line, 34, "F2", (0.98, 0.99, 1.0))
        y += 43
    y += 22
    for line in wrap(meta[0] if meta else "System architecture, models, data, setup, validation, and limitations", 58):
        c.text(56, y, line, 13, "F1", (0.76, 0.81, 0.91))
        y += 20
    y += 48
    c.rect(54, y, 485, 1, (0.25, 0.31, 0.47))
    y += 28
    for item in meta[1:]:
        for line in wrap(item, 69):
            c.text(56, y, line, 9.5, "F1", (0.66, 0.72, 0.84))
            y += 17
    c.text(56, 758, "A plain-language guide to the code as it is implemented today.", 10, "F1", (0.69, 0.75, 0.88))
    c.text(540, 808, "01", 9, "F2", (0.69, 0.75, 0.88))
    return c


def draw_toc(chapters, chapter_pages):
    c = Canvas()
    c.rect(0, 0, PAGE_W, PAGE_H, (1, 1, 1))
    c.text(LEFT, 38, "DEEPSIGNAL  /  PROJECT GUIDE", 8, "F2", MUTED)
    c.text(PAGE_W-RIGHT-150, 38, "CONTENTS", 7.2, "F1", MUTED)
    c.line(LEFT, 48, PAGE_W-RIGHT, 48, LINE, 0.6)
    c.text(LEFT, 104, "Contents", 24, "F2", INK)
    c.text(LEFT, 126, "Use this map to jump to any part of the guide.", 10, "F1", MUTED)
    c.line(LEFT, 142, PAGE_W-RIGHT, 142, LINE, 0.8)
    if len(chapters) <= 21:
        y = 171
        for idx, chapter in enumerate(chapters, start=1):
            c.circle(LEFT+7, y-3, 6.5, BLUE)
            c.text(LEFT+4.1, y, f"{idx:02d}", 6.5, "F2", (1, 1, 1))
            c.text(LEFT+23, y, chapter, 9.4, "F1", INK)
            c.text(PAGE_W-RIGHT-28, y, str(chapter_pages.get(chapter, "")), 9, "F2", MUTED)
            c.line(LEFT+23, y+8, PAGE_W-RIGHT, y+8, (0.92, 0.93, 0.95), 0.45)
            y += 28
    else:
        # Two compact columns keep the contents readable as the guide expands.
        rows = (len(chapters) + 1) // 2
        col_width = CONTENT_W / 2
        for idx, chapter in enumerate(chapters):
            col = idx // rows
            row = idx % rows
            x = LEFT + col * col_width
            y = 171 + row * 42
            c.circle(x+7, y-3, 6.5, BLUE)
            c.text(x+4.1, y, f"{idx+1:02d}", 6.5, "F2", (1, 1, 1))
            title_lines = wrap(chapter, 28)
            for line_idx, line in enumerate(title_lines[:2]):
                c.text(x+23, y+line_idx*10, line, 7.4, "F1", INK)
            c.text(x+col_width-20, y, str(chapter_pages.get(chapter, "")), 8, "F2", MUTED)
            c.line(x+23, y+23, x+col_width-12, y+23, (0.92, 0.93, 0.95), 0.45)
    c.line(LEFT, 800, PAGE_W-RIGHT, 800, LINE, 0.6)
    c.text(LEFT, 819, "02", 9, "F2", MUTED)
    return c


def render_body(blocks):
    pages = []
    c = add_page(pages)
    chapter_pages = {}
    chapter_num = 0
    for kind, value in blocks:
        if kind == "chapter":
            if c.has_content:
                c = add_page(pages)
            chapter_num += 1
            chapter_pages[value] = len(pages) + 2  # cover + contents pages
            lines = wrap(value, 48)
            c.rect(LEFT, c.y-20, 4, 31, BLUE)
            c.text(LEFT+14, c.y, value, 17, "F2", INK)
            c.y += max(28, len(lines)*21) + 5
            c.line(LEFT, c.y-6, PAGE_W-RIGHT, c.y-6, LINE, 0.8)
            c.y += 10
            c.has_content = True
        elif kind == "sub":
            c = ensure_space(c, pages, 40)
            c.y += 6
            c.text(LEFT, c.y, value, 12, "F2", BLUE)
            c.y += 18
            c.has_content = True
        elif kind == "p":
            lines = wrap(value, 93)
            c = ensure_space(c, pages, len(lines)*13.2 + 9)
            for line in lines:
                c.text(LEFT, c.y, line, 9.25, "F1", INK)
                c.y += 13.2
            c.y += 6
            c.has_content = True
        elif kind == "bullet":
            lines = wrap(value, 87)
            c = ensure_space(c, pages, len(lines)*13.2 + 5)
            c.circle(LEFT+3, c.y-3, 2, BLUE)
            for idx, line in enumerate(lines):
                c.text(LEFT+13, c.y, line, 9.15, "F1", INK)
                c.y += 13.2
            c.y += 3
            c.has_content = True
        elif kind == "callout":
            lines = wrap(value, 82)
            h = len(lines)*14 + 22
            c = ensure_space(c, pages, h+10)
            c.rect(LEFT, c.y-12, CONTENT_W, h, PALE, (0.82, 0.87, 0.94), 0.7)
            c.rect(LEFT, c.y-12, 3, h, BLUE)
            y = c.y+5
            for line in lines:
                c.text(LEFT+14, y, line, 9.1, "F1", INK)
                y += 14
            c.y += h
            c.has_content = True
        elif kind == "code":
            max_chars = 99
            rows = []
            for row in value:
                rows.extend(wrap(row, max_chars) if row else [""])
            h = len(rows)*10.8 + 20
            c = ensure_space(c, pages, min(h, BOTTOM-TOP))
            # Split long code blocks over pages without losing indentation.
            for start in range(0, len(rows), 54):
                chunk = rows[start:start+54]
                chunk_h = len(chunk)*10.8+18
                c = ensure_space(c, pages, chunk_h)
                c.rect(LEFT, c.y-13, CONTENT_W, chunk_h, (0.955, 0.965, 0.982), (0.88, 0.90, 0.94), 0.5)
                y = c.y+2
                for line in chunk:
                    c.text(LEFT+10, y, line, 7.25, "F3", (0.18, 0.24, 0.36))
                    y += 10.8
                c.y += chunk_h+8
                c.has_content = True
                if start+54 < len(rows):
                    c = add_page(pages)
    return pages, chapter_pages


def make_pdf(page_canvases, path: Path):
    objects: list[bytes | None] = [None] * 6
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    objects[4] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
    objects[5] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>"
    page_ids = []
    for canvas in page_canvases:
        content = "\n".join(canvas.commands).encode("ascii")
        content_id = len(objects)
        objects.append(f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"\nendstream")
        page_id = len(objects)
        page_ids.append(page_id)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W:.2f} {PAGE_H:.2f}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode("ascii")
        )
    kids = " ".join(f"{obj_id} 0 R" for obj_id in page_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
    objects.append(b"<< /Title (DeepSignal Project Guide) /Author (DeepSignal Project) /Subject (Architecture, models, setup, and validation) >>")
    info_id = len(objects)-1
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0] * len(objects)
    for obj_id in range(1, len(objects)):
        offsets[obj_id] = len(output)
        output.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        output.extend(objects[obj_id] or b"")
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects)}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for obj_id in range(1, len(objects)):
        output.extend(f"{offsets[obj_id]:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects)} /Root 1 0 R /Info {info_id} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(output)


def main():
    title, meta, chapters, blocks = split_markdown(SOURCE.read_text(encoding="utf-8"))
    body_pages, chapter_pages = render_body(blocks)
    cover = draw_cover(title, meta)
    toc = draw_toc(chapters, chapter_pages)
    all_pages = [cover, toc, *body_pages]
    for number, canvas in enumerate(all_pages, start=1):
        if number > 2:
            canvas.text(PAGE_W-RIGHT-24, 819, f"{number:02d}", 9, "F2", MUTED)
    make_pdf(all_pages, OUTPUT)
    print(f"Created {OUTPUT} ({len(all_pages)} pages)")
    print(f"Table of contents entries: {len(chapter_pages)}")


if __name__ == "__main__":
    main()
