"""Render report/report.md to a paginated PDF; Markdown remains the source of truth."""
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                               Table, TableStyle, Image, KeepTogether)
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
import matplotlib

ROOT = Path(__file__).resolve().parents[1]
WIDTH = A4[0] - 88
NAVY = colors.HexColor('#19324b')
TEAL = colors.HexColor('#176c79')
BODY = colors.HexColor('#253445')
REPOSITORY = 'https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15/blob/main/'
FONT_DIR = Path(matplotlib.get_data_path()) / 'fonts/ttf'
if FONT_DIR.is_dir():
    for name, filename in [('ReportSans', 'DejaVuSans.ttf'),
                           ('ReportSans-Bold', 'DejaVuSans-Bold.ttf'),
                           ('ReportSans-Italic', 'DejaVuSans-Oblique.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
    pdfmetrics.registerFontFamily('ReportSans', normal='ReportSans', bold='ReportSans-Bold',
                                  italic='ReportSans-Italic', boldItalic='ReportSans-Bold')
styles = {
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=9.0, leading=12.2,
                           textColor=BODY, spaceAfter=6),
    'h1': ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=29, leading=33,
                         textColor=NAVY, spaceBefore=28, spaceAfter=21),
    'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=15.1, leading=18.5,
                         textColor=NAVY, spaceBefore=8, spaceAfter=9, keepWithNext=True),
    'h3': ParagraphStyle('h3', fontName='Helvetica-Bold', fontSize=10.7, leading=14,
                         textColor=TEAL, spaceBefore=6, spaceAfter=5, keepWithNext=True),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=8.0, leading=10.3,
                           textColor=BODY),
    'caption': ParagraphStyle('caption', fontName='Helvetica-Oblique', fontSize=8.1,
                              leading=10.3, textColor=TEAL, spaceAfter=6),
    'bullet': ParagraphStyle('bullet', fontName='Helvetica', fontSize=9.0, leading=12.2,
                             leftIndent=10, firstLineIndent=0, bulletIndent=0,
                             textColor=BODY, spaceAfter=5),
}
if FONT_DIR.is_dir():
    for style in styles.values():
        style.fontName = {'Helvetica': 'ReportSans', 'Helvetica-Bold': 'ReportSans-Bold',
                          'Helvetica-Oblique': 'ReportSans-Italic'}[style.fontName]


def inline(text):
    text = html.escape(text)
    def link(match):
        label, target = match.groups()
        if not target.startswith(('https://', 'http://')):
            resolved = (ROOT / 'report' / html.unescape(target)).resolve()
            target = REPOSITORY + resolved.relative_to(ROOT).as_posix()
        return f'<link href="{target}" color="#176c79">{label}</link>'
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    return text


def table_block(lines):
    raw = [[cell.strip() for cell in line.strip().strip('|').split('|')] for line in lines]
    rows = [row for row in raw if not all(re.fullmatch(r'[:\- ]+', x) for x in row)]
    cols = len(rows[0])
    assert all(len(r) == cols for r in rows)
    lengths = [max(len(r[i]) for r in rows) for i in range(cols)]
    if cols >= 5:
        widths = [WIDTH * .31] + [WIDTH * .69 / (cols - 1)] * (cols - 1)
    elif cols == 2:
        widths = [WIDTH * .34, WIDTH * .66]
    elif cols == 3 and 'Student ID' in rows[0]:
        widths = [WIDTH * .27, WIDTH * .16, WIDTH * .57]
    else:
        weights = [max(8, min(60, n)) ** .68 for n in lengths]
        widths = [WIDTH * w / sum(weights) for w in weights]
    data = [[Paragraph(('<b>' + inline(c) + '</b>') if i == 0 else inline(c), styles['cell'])
             for c in row] for i, row in enumerate(rows)]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e7eff3')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f8fa')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#acc3ce')),
        ('LINEBELOW', (0, -1), (-1, -1), .35, colors.HexColor('#cad8df')),
    ]))
    return [table, Spacer(1, 7)]


def parse(text):
    story = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak()); i += 1; continue
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            story.append(Paragraph(inline(line[level:].strip()), styles['h' + str(level)]))
            i += 1; continue
        image_match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if image_match:
            caption, relative = image_match.groups()
            path = (ROOT / 'report' / relative).resolve()
            w, h = ImageReader(str(path)).getSize()
            width = WIDTH
            height = width * h / w
            max_height = 255 if path.name.startswith('11_') else 194
            if height > max_height:
                width *= max_height / height; height = max_height
            story.append(KeepTogether([Image(str(path), width=width, height=height),
                                       Spacer(1, 4), Paragraph(inline(caption), styles['caption'])]))
            i += 1; continue
        if line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                block.append(lines[i]); i += 1
            story.extend(table_block(block)); continue
        if line.startswith('- '):
            story.append(Paragraph(inline(line[2:]), styles['bullet'], bulletText='-'))
            i += 1; continue
        if re.match(r'^\d+\. ', line):
            story.append(Paragraph(inline(line), styles['body']))
            i += 1; continue
        paragraph = [line]; i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', '![', '<!--', '- ')) and not re.match(r'^\d+\. ', lines[i]):
            paragraph.append(lines[i].strip()); i += 1
        story.append(Paragraph(inline(' '.join(paragraph)), styles['body']))
    return story


def page_chrome(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#c7d6df'))
    canvas.line(44, A4[1] - 35, A4[0] - 44, A4[1] - 35)
    canvas.setFont('ReportSans' if FONT_DIR.is_dir() else 'Helvetica', 7)
    canvas.setFillColor(TEAL)
    canvas.drawString(44, A4[1] - 26, 'CSE437  /  DATA SCIENCE  /  GROUP 15')
    canvas.drawRightString(A4[0] - 44, A4[1] - 26, 'SECTION 05  |  SUMMER 2026')
    canvas.setFillColor(colors.HexColor('#647487'))
    canvas.drawString(44, 26, 'Technical report - final declaration and submission review pending')
    canvas.drawRightString(A4[0] - 44, 26, str(doc.page))
    canvas.restoreState()


def main():
    text = (ROOT / 'report/report.md').read_text()
    summary = text.split('## Summary\n\n')[1].split('\n\n**Submission status:')[0]
    assert 150 <= len(summary.split()) <= 200
    assert text.count('<!-- pagebreak -->') == 9
    output = ROOT / 'report/report.pdf'
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=44, leftMargin=44,
                            topMargin=47, bottomMargin=44, title='Hotel Booking Cancellation Project',
                            author='CSE437 Group 15', pageCompression=1)
    doc.build(parse(text), onFirstPage=page_chrome, onLaterPages=page_chrome)
    pages = len(PdfReader(output).pages)
    print(f'{output}: {pages} pages; summary {len(summary.split())} words')
    if pages != 10:
        raise RuntimeError(f'Expected exactly 10 pages; found {pages}. Review layout before delivery.')


if __name__ == '__main__':
    main()
