"""Build the DeepSignal Markdown guide as a dependency-free, searchable PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "DeepSignal_Project_Guide.md"
OUTPUT = ROOT / "docs" / "DeepSignal_Project_Guide.pdf"
PAGE_W, PAGE_H = 595.28, 841.89  # A4, points
LEFT, RIGHT = 48.0, 48.0
TOP, BOTTOM = 795.0, 48.0
CONTENT_W = PAGE_W - LEFT - RIGHT


def plain(text: str) -> str:
    """Remove Markdown inline syntax while keeping link labels readable."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("__", "")
    text = text.replace("`", "").replace("*", "")
    text = html.unescape(text)
    return (text.replace("→", "->").replace("←", "<- ")
            .replace("↔", "<->").replace("—", "- ").replace("–", "-")
            .replace("≥", ">=").replace("≤", "<=").replace("±", "+/-"))


def text_width(text: str, size: float, mono: bool = False) -> float:
    if mono:
        return len(text) * size * 0.60
    narrow = set("il.,:;!|'`[](){}")
    wide = set("MW@%&QG")
    total = 0.0
    for char in text:
        if char == " ":
            factor = 0.27
        elif char in narrow:
            factor = 0.26
        elif char in wide:
            factor = 0.78
        elif char.isupper():
            factor = 0.64
        elif char.isdigit():
            factor = 0.55
        else:
            factor = 0.50
        total += size * factor
    return total


def wrap(text: str, width: float, size: float, mono: bool = False) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines, current = [], words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_width(candidate, size, mono) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def parse_markdown(source: str) -> list[tuple]:
    lines = source.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):
            language = stripped[3:].strip().lower()
            i += 1
            content = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                content.append(lines[i])
                i += 1
            i += 1
            blocks.append(("diagram",) if language == "mermaid" else ("code", "\n".join(content)))
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            blocks.append(("heading", len(heading.group(1)), plain(heading.group(2))))
            i += 1
            continue
        if stripped == "---":
            blocks.append(("rule",))
            i += 1
            continue
        if stripped.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(plain(lines[i].strip()[1:].strip()))
                i += 1
            blocks.append(("quote", " ".join(quote)))
            continue
        bullet = re.match(r"^[-*]\s+(.+)$", stripped)
        numbered = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if bullet or numbered:
            blocks.append(("list", "bullet" if bullet else "number", plain((bullet or numbered).group(1))))
            i += 1
            continue
        paragraph = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", "```", "|", ">", "---"))
                    or re.match(r"^(?:[-*]\s+|\d+\.\s+)", nxt)):
                break
            paragraph.append(nxt)
            i += 1
        blocks.append(("paragraph", plain(" ".join(paragraph))))
    return blocks


class PDFDocument:
    def __init__(self):
        self.pages: list[list[str]] = []
        self.page = []
        self.y = TOP
        self.page_number = 0
        self.new_page()

    @staticmethod
    def _color(rgb):
        return " ".join(f"{v / 255:.3f}" for v in rgb)

    @staticmethod
    def _safe_bytes(value: str) -> bytes:
        return value.encode("cp1252", errors="replace")

    @classmethod
    def _pdf_string(cls, value: str) -> str:
        return "<" + cls._safe_bytes(value).hex().upper() + ">"

    def new_page(self):
        if self.page:
            self.pages.append(self.page)
        self.page_number += 1
        self.page = []
        self.y = TOP
        self.page.append("0.91 0.93 0.96 RG 48 815 m 547 815 l S")
        self.text(48, 823, "DeepSignal  /  Complete Project Guide", 7.5, "F1", (93, 108, 128))

    def ensure(self, height: float):
        if self.y - height < BOTTOM:
            self.new_page()

    def text(self, x, y, value, size=9.2, font="F1", color=(37, 50, 72)):
        self.page.append(
            f"BT /{font} {size:.2f} Tf {self._color(color)} rg 1 0 0 1 {x:.2f} {y:.2f} Tm "
            f"{self._pdf_string(value)} Tj ET"
        )

    def line(self, x1, y1, x2, y2, color=(205, 216, 232), width=0.7):
        self.page.append(f"{self._color(color)} RG {width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def rect(self, x, y, width, height, fill, stroke=None, line_width=1):
        self.page.append(
            f"q {self._color(fill)} rg "
            + (f"{self._color(stroke)} RG {line_width:.2f} w " if stroke else "")
            + f"{x:.2f} {y:.2f} {width:.2f} {height:.2f} re "
            + ("B Q" if stroke else "f Q")
        )

    def paragraph(self, value, size=9.2, color=(37, 50, 72), indent=0, leading=None):
        leading = leading or size * 1.42
        lines = wrap(value, CONTENT_W - indent, size)
        self.ensure(len(lines) * leading + 8)
        for line in lines:
            self.text(LEFT + indent, self.y, line, size, "F1", color)
            self.y -= leading
        self.y -= 5

    def heading(self, level, value):
        if level == 1:
            if self.page_number > 1 or self.y < TOP - 20:
                self.new_page()
            self.y -= 30
            size, font, color, gap = 24, "F2", (19, 49, 85), 12
        elif level == 2:
            self.ensure(40)
            self.y -= 11
            size, font, color, gap = 15, "F2", (27, 67, 117), 7
        else:
            self.ensure(30)
            self.y -= 7
            size, font, color, gap = 11.2, "F2", (39, 88, 139), 5
        lines = wrap(value, CONTENT_W, size)
        for line in lines:
            self.text(LEFT, self.y, line, size, font, color)
            self.y -= size * 1.25
        if level == 2:
            self.line(LEFT, self.y + 1, PAGE_W - RIGHT, self.y + 1, (200, 214, 232), 1)
        self.y -= gap

    def list_item(self, kind, value, number):
        size, leading, indent = 8.8, 12.2, 13
        lines = wrap(value, CONTENT_W - indent, size)
        self.ensure(len(lines) * leading + 2)
        prefix = "•" if kind == "bullet" else f"{number}."
        self.text(LEFT + 1, self.y, prefix, size, "F2", (37, 91, 148))
        for index, line in enumerate(lines):
            self.text(LEFT + indent, self.y, line, size, "F1")
            self.y -= leading
        self.y -= 1.5

    def quote(self, value):
        size, leading, pad = 9.0, 13.0, 9
        lines = wrap(value, CONTENT_W - 2 * pad, size)
        height = len(lines) * leading + 2 * pad
        self.ensure(height + 7)
        bottom = self.y - height
        self.rect(LEFT, bottom, CONTENT_W, height, (237, 245, 253))
        self.rect(LEFT, bottom, 3, height, (45, 120, 189))
        for index, line in enumerate(lines):
            self.text(LEFT + pad + 3, self.y - pad - index * leading, line, size, "F1", (32, 59, 92))
        self.y = bottom - 8

    def code(self, value):
        raw_lines = value.splitlines() or [""]
        size, leading, pad = 7.1, 9.4, 7
        code_lines = []
        for source_line in raw_lines:
            code_lines.extend(wrap(source_line, CONTENT_W - 2 * pad, size, mono=True))
        height = len(code_lines) * leading + 2 * pad
        if height > PAGE_H - TOP + BOTTOM:
            for source_line in code_lines:
                self.ensure(leading + 2 * pad)
                self.rect(LEFT, self.y - leading - 2 * pad, CONTENT_W, leading + 2 * pad, (241, 244, 248))
                self.text(LEFT + pad, self.y - pad, source_line, size, "F4", (38, 55, 77))
                self.y -= leading + 2 * pad
            self.y -= 6
            return
        self.ensure(height + 6)
        self.rect(LEFT, self.y - height, CONTENT_W, height, (241, 244, 248))
        self.rect(LEFT, self.y - height, 3, height, (65, 125, 184))
        for index, line in enumerate(code_lines):
            self.text(LEFT + pad + 3, self.y - pad - index * leading, line, size, "F4", (38, 55, 77))
        self.y -= height + 7

    def table(self, rows):
        if not rows:
            return
        columns = max(len(row) for row in rows)
        proportions = {
            2: [0.29, 0.71],
            3: [0.22, 0.43, 0.35],
            4: [0.22, 0.28, 0.25, 0.25],
        }.get(columns, [1 / columns] * columns)
        widths = [CONTENT_W * p for p in proportions]
        size, leading, pad = 7.6, 10.1, 4.5
        for row_index, row in enumerate(rows):
            wrapped = []
            for col in range(columns):
                cell = plain(row[col]) if col < len(row) else ""
                wrapped.append(wrap(cell, widths[col] - 2 * pad, size))
            height = max(1, max(len(lines) for lines in wrapped)) * leading + 2 * pad
            if self.y - height < BOTTOM:
                self.new_page()
                if row_index > 0:
                    self.y -= 3
            bottom = self.y - height
            self.rect(LEFT, bottom, CONTENT_W, height,
                      (23, 62, 112) if row_index == 0 else ((244, 247, 251) if row_index % 2 == 0 else (255, 255, 255)),
                      (196, 208, 224), 0.45)
            x = LEFT
            for col, cell_lines in enumerate(wrapped):
                if col:
                    self.line(x, bottom, x, self.y, (196, 208, 224), 0.45)
                for line_index, line in enumerate(cell_lines):
                    color = (255, 255, 255) if row_index == 0 else (37, 50, 72)
                    font = "F2" if row_index == 0 or (col == 0 and columns > 2) else "F1"
                    self.text(x + pad, self.y - pad - size * 0.8 - line_index * leading,
                              line, size, font, color)
                x += widths[col]
            self.y = bottom
        self.y -= 8

    def rule(self):
        self.ensure(16)
        self.y -= 4
        self.line(LEFT, self.y, PAGE_W - RIGHT, self.y, (201, 212, 227), 0.8)
        self.y -= 12

    def arrow(self, x1, y1, x2, y2, color=(76, 96, 121)):
        self.line(x1, y1, x2, y2, color, 1.5)
        if abs(x2 - x1) < 2:
            direction = -1 if y2 < y1 else 1
            self.page.append(
                f"q {self._color(color)} rg {x2-4:.2f} {y2+direction*2:.2f} m "
                f"{x2+4:.2f} {y2+direction*2:.2f} l {x2:.2f} {y2-direction*5:.2f} l f Q"
            )
        else:
            direction = 1 if x2 > x1 else -1
            self.page.append(
                f"q {self._color(color)} rg {x2-direction*5:.2f} {y2+4:.2f} m "
                f"{x2-direction*5:.2f} {y2-4:.2f} l {x2+direction*2:.2f} {y2:.2f} l f Q"
            )

    def diagram_box(self, x, top, width, height, fill, stroke, title, detail=(), title_color=(25, 44, 69), tag=None):
        self.rect(x, top - height, width, height, fill, stroke, 1.7)
        center = x + width / 2
        if tag:
            self.text(center - text_width(tag, 7.0) / 2, top - 12, tag, 7.0, "F2", stroke)
            title_y = top - 27
        else:
            title_y = top - 17
        self.text(center - text_width(title, 9.0) / 2, title_y, title, 9.0, "F2", title_color)
        if detail:
            detail_lines = wrap(detail, width - 14, 7.0)
            for i, line in enumerate(detail_lines[:2]):
                self.text(center - text_width(line, 7.0) / 2, title_y - 12 - i * 9, line, 7.0, "F1", (57, 72, 94))

    def architecture(self):
        self.new_page()
        self.heading(3, "System architecture — model components highlighted")
        y = self.y
        width = CONTENT_W
        x = LEFT
        self.diagram_box(x + 30, y, width - 60, 43, (246, 248, 252), (105, 122, 145),
                         "Browser · React application", "Ticker picker · dashboard · analysis · charts · local saved signals")
        self.arrow(PAGE_W / 2, y - 43, PAGE_W / 2, y - 56)
        y -= 60
        self.diagram_box(x + 65, y, width - 130, 43, (246, 248, 252), (105, 122, 145),
                         "Vite → FastAPI", "HTTP / JSON · frontend :5173 · backend :8000")
        self.arrow(PAGE_W / 2, y - 43, PAGE_W / 2, y - 56)
        y -= 60
        self.diagram_box(x + 18, y, width - 36, 52, (246, 248, 252), (105, 122, 145),
                         "Data providers and features", "Yahoo Finance daily OHLCV · NewsAPI / Yahoo headlines · pandas features and labels")
        self.arrow(PAGE_W / 2, y - 52, PAGE_W / 2, y - 66)
        y -= 70

        group_h = 190
        self.rect(x + 8, y - group_h, width - 16, group_h, (250, 252, 255), (160, 174, 194), 1.2)
        self.text(PAGE_W / 2 - 119, y - 16, "MODEL COMPONENTS + DIRECTION EVIDENCE", 8.0, "F2", (56, 76, 101))
        gap = 12
        card_w = (width - 50 - gap) / 2
        left_x, right_x = x + 25, x + 25 + card_w + gap
        card_h = 54
        first_top, second_top = y - 31, y - 96
        blue = (42, 117, 196)
        model_fill = (229, 241, 255)
        self.diagram_box(left_x, first_top, card_w, card_h, model_fill, blue, "LSTM", "Sequence model · next-session up · 40%", (22, 62, 107), "MODEL · SEQUENCE")
        self.diagram_box(right_x, first_top, card_w, card_h, model_fill, blue, "Random Forest", "Tabular model · next-session up · 40%", (22, 62, 107), "MODEL · TABULAR")
        self.diagram_box(left_x, second_top, card_w, card_h, model_fill, blue, "FinBERT", "Headline sentiment mapped to score · 10%", (22, 62, 107), "MODEL · TEXT")
        self.diagram_box(right_x, second_top, card_w, card_h, (245, 247, 251), (110, 129, 154), "Technical evidence", "Moving averages · momentum · RSI/MACD · 10%", (37, 57, 81), "EXPLICIT RULES")
        self.arrow(PAGE_W / 2, y, PAGE_W / 2, y - 24)
        hybrid_top = y - group_h - 17
        self.arrow(PAGE_W / 2, y - group_h, PAGE_W / 2, hybrid_top - 2)
        hybrid_w = width - 90
        hybrid_x = x + (width - hybrid_w) / 2
        self.diagram_box(hybrid_x, hybrid_top, hybrid_w, 48, (222, 247, 236), (23, 133, 104),
                         "Weighted hybrid → Buy / Hold / Sell", "Versioned score · 40% LSTM · 40% RF · 10% sentiment · 10% technical",
                         (18, 105, 81), "ENSEMBLE OUTPUT")
        y = hybrid_top - 64
        half = (width - 18) / 2
        self.diagram_box(x, y, half, 54, (244, 236, 252), (126, 82, 174), "XGBoost (optional)", "Holdout comparison only; not a hybrid voter", (94, 58, 133), "COMPARISON MODEL")
        self.diagram_box(x + half + 18, y, half, 54, (248, 246, 239), (138, 120, 82), "Backtest and persistence", "Next-open rules · Supabase / JSON · local model artifacts", (91, 77, 49))
        self.arrow(hybrid_x + hybrid_w / 2, hybrid_top - 48, hybrid_x + hybrid_w / 2, y + 5)
        self.arrow(x + half, y - 27, x + half + 18, y - 27)
        cap = "Blue = trained model component   Green = explicit hybrid   Purple = optional comparison"
        self.text(LEFT + 14, y - 68, cap, 7.3, "F1", (78, 93, 113))
        self.y = y - 82

    def add_block(self, block, number):
        kind = block[0]
        if kind == "heading":
            self.heading(block[1], block[2])
        elif kind == "paragraph":
            self.paragraph(block[1])
        elif kind == "list":
            self.list_item(block[1], block[2], number)
        elif kind == "quote":
            self.quote(block[1])
        elif kind == "code":
            self.code(block[1])
        elif kind == "table":
            self.table(block[1])
        elif kind == "rule":
            self.rule()
        elif kind == "diagram":
            self.architecture()

    def save(self, path: Path):
        self.pages.append(self.page)
        objects: dict[int, bytes] = {}
        objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
        objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
        objects[4] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
        objects[5] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique /Encoding /WinAnsiEncoding >>"
        objects[6] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>"
        page_ids, next_id = [], 7
        for page_number, commands in enumerate(self.pages, start=1):
            commands = commands + [
                f"BT /F1 7.5 Tf 0.35 0.39 0.45 rg 1 0 0 1 48 25 Tm {self._pdf_string('DeepSignal Project Guide')} Tj ET",
                f"BT /F1 7.5 Tf 0.35 0.39 0.45 rg 1 0 0 1 520 25 Tm {self._pdf_string(f'Page {page_number}')} Tj ET",
            ]
            stream = "\n".join(commands).encode("ascii")
            content_id, page_id = next_id, next_id + 1
            next_id += 2
            objects[content_id] = (
                f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
                + stream + b"\nendstream"
            )
            objects[page_id] = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W:.2f} {PAGE_H:.2f}] "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R /F4 6 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
            page_ids.append(page_id)
        kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
        objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
        max_id = max(objects)
        output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0] * (max_id + 1)
        for object_id in range(1, max_id + 1):
            offsets[object_id] = len(output)
            output.extend(f"{object_id} 0 obj\n".encode("ascii"))
            output.extend(objects[object_id])
            output.extend(b"\nendobj\n")
        xref_position = len(output)
        output.extend(f"xref\n0 {max_id + 1}\n".encode("ascii"))
        output.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        output.extend(
            f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref_position}\n%%EOF\n".encode("ascii")
        )
        path.write_bytes(output)


def main():
    if not SOURCE.is_file():
        raise SystemExit(f"Guide not found: {SOURCE}")
    document = PDFDocument()
    blocks = parse_markdown(SOURCE.read_text(encoding="utf-8"))
    list_number = 0
    for block in blocks:
        if block[0] == "list" and block[1] == "number":
            list_number += 1
        else:
            list_number = 0
        document.add_block(block, list_number)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes, {len(document.pages)} pages)")


if __name__ == "__main__":
    main()
