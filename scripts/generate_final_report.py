#!/usr/bin/env python3
"""Generate Markdown, DOCX, and PDF reports from real SynthEx run artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_PAGES = 20
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\malgun.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_font() -> Path | None:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def build_blocks(args, manifest, report, strategies, seed) -> list[dict[str, Any]]:
    failure = manifest.get("failure") or {}
    usage = manifest.get("llm_usage") or {}
    durations = manifest.get("stage_durations_seconds") or {}
    status = manifest.get("status", "unknown")
    llm_cfg = (manifest.get("config", {}).get("llm", {}) or {})

    blocks: list[dict[str, Any]] = [
        {"type": "title", "text": args.title or "SynthEx + Solar Pro 4 실행 보고서"},
        {"type": "meta", "text": f"생성일: {datetime.now(timezone.utc).isoformat()} | run_id: {manifest.get('run_id', 'unknown')}"},
        {"type": "callout", "text": "본 보고서는 실제 저장된 실행 기록만 근거로 생성된다. 보고서 생성 성공은 SynthEx 합성 경로 탐색 성공을 의미하지 않는다."},
        {"type": "h1", "text": "1. 실행 요약"},
        {"type": "table", "headers": ["항목", "기록된 값"], "rows": [
            ["대상 SMILES", manifest.get("target", "미확보")],
            ["target_id", manifest.get("target_id", "미확보")],
            ["상태", status],
            ["현재 단계", manifest.get("current_stage", "미확보")],
            ["완료 단계", ", ".join(manifest.get("completed_stages") or []) or "없음"],
            ["요청 모델", llm_cfg.get("model", "미확보")],
            ["실제 provider", report.get("provider", "미확보")],
        ]},
        {"type": "h1", "text": "2. 단계별 결과"},
        {"type": "table", "headers": ["단계", "상태", "시간(초)"], "rows": [
            ["strategies", "완료" if "strategies" in (manifest.get("completed_stages") or []) else "미완료", f"{durations.get('strategies', 0):.6f}"],
            ["routes", "실패" if status == "failed" and manifest.get("current_stage") == "routes" else "기록 확인 필요", f"{durations.get('routes', 0):.6f}"],
            ["reviewed_routes", "미완료", "-"],
            ["analyses", "미완료", "-"],
            ["stitched_routes", "미완료", "-"],
        ]},
        {"type": "h1", "text": "3. 오류 기록"},
        {"type": "table", "headers": ["항목", "내용"], "rows": [
            ["실패 단계", failure.get("stage", "미확보")],
            ["범주", failure.get("category", "미확보")],
            ["메시지", failure.get("message", "기록 없음")],
        ]},
        {"type": "h1", "text": "4. 실제 생성된 전략"},
    ]
    for idx, strategy in enumerate(strategies.get("strategies") or [], 1):
        blocks.append({"type": "p", "text": f"{idx}. {strategy}"})
    if not strategies.get("strategies"):
        blocks.append({"type": "p", "text": "기록된 전략 없음"})

    blocks += [
        {"type": "h1", "text": "5. LLM 사용량"},
        {"type": "table", "headers": ["항목", "값"], "rows": [
            ["호출 수", str(usage.get("calls", "미확보"))],
            ["입력 토큰", str(usage.get("input_tokens", "미확보"))],
            ["출력 토큰", str(usage.get("output_tokens", "미확보"))],
            ["총 토큰", str(usage.get("total_tokens", "미확보"))],
        ]},
        {"type": "h1", "text": "6. 합성 조건 및 수율"},
        {"type": "p", "text": "온도, 시간, 용매, 촉매, 수율: 미확보. 실제 검증된 경로 기록에 값이 있을 때만 기재한다."},
        {"type": "h1", "text": "7. 합성 구조 이미지"},
        {"type": "p", "text": "실제 확보된 합성 경로가 없으므로 구조 이미지를 포함하지 않는다."},
        {"type": "h1", "text": "8. 화학·실험 검증 구분"},
        {"type": "p", "text": "본 기록은 소프트웨어 실행 결과다. 화학적 타당성 검증 및 실험적 검증은 수행 기록이 없으면 미확보로 처리한다."},
    ]
    if seed:
        seed_failure = seed.get("failure") or {}
        blocks += [
            {"type": "h1", "text": "9. 별도 후속 실험 기록"},
            {"type": "p", "text": f"run_id {seed.get('run_id', 'unknown')}: {seed_failure.get('message', '기록 없음')}"},
        ]
    blocks += [
        {"type": "h1", "text": "10. 판정"},
        {"type": "callout", "text": f"최종 상태: {status}. 실제 최종 합성 경로는 {'확보되지 않음' if status != 'completed' else '체크포인트 검증 필요'}."},
    ]
    return blocks


def render_markdown(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for block in blocks:
        kind = block["type"]
        text = block.get("text", "").replace("\n", "<br>")
        if kind == "title":
            lines.append("# " + text)
        elif kind in ("h1", "h2"):
            lines.append("## " + text)
        elif kind in ("callout", "note"):
            lines.append("> " + text)
        elif kind == "table":
            headers = block["headers"]
            lines.append("| " + " | ".join(str(h) for h in headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in block["rows"]:
                lines.append("| " + " | ".join(str(c).replace("\n", " ") for c in row) + " |")
        else:
            lines.append(text)
        lines.append("")
    return "\n".join(lines)


def render_docx(blocks: list[dict[str, Any]], path: Path) -> None:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt

    doc = Document()
    font_name = "Malgun Gothic"
    for style_name in ("Normal", "Title", "Heading 1", "Heading 2"):
        style = doc.styles[style_name]
        style.font.name = font_name
        rpr = style.element.get_or_add_rPr()
        rfonts = rpr.get_or_add_rFonts()
        rfonts.set(qn("w:eastAsia"), font_name)

    for block in blocks:
        kind = block["type"]
        if kind == "table":
            table = doc.add_table(rows=1, cols=len(block["headers"]))
            table.style = "Table Grid"
            for idx, header in enumerate(block["headers"]):
                table.rows[0].cells[idx].text = str(header)
            for row in block["rows"]:
                cells = table.add_row().cells
                for idx, value in enumerate(row):
                    cells[idx].text = str(value)
            continue
        para = doc.add_paragraph()
        if kind == "title":
            para.style = doc.styles["Title"]
        elif kind == "h1":
            para.style = doc.styles["Heading 1"]
        run = para.add_run(block.get("text", ""))
        run.font.name = font_name
        run.font.size = Pt(10)
        rpr = run._element.get_or_add_rPr()
        rfonts = rpr.get_or_add_rFonts()
        rfonts.set(qn("w:eastAsia"), font_name)
    doc.save(path)


def render_pdf(blocks: list[dict[str, Any]], path: Path) -> int:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font_path = find_font()
    base_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    if font_path:
        pdfmetrics.registerFont(TTFont("ReportFont", str(font_path)))
        base_font = bold_font = "ReportFont"
    else:
        print("WARNING: No Korean TTF found; PDF may not render Korean.", file=sys.stderr)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="RTitle", fontName=bold_font, fontSize=17, leading=22, wordWrap="CJK", alignment=TA_LEFT, spaceAfter=10))
    styles.add(ParagraphStyle(name="RH1", fontName=bold_font, fontSize=12, leading=16, wordWrap="CJK", spaceBefore=10, spaceAfter=5))
    styles.add(ParagraphStyle(name="RP", fontName=base_font, fontSize=9, leading=13, wordWrap="CJK", spaceAfter=4))
    styles.add(ParagraphStyle(name="RCell", fontName=base_font, fontSize=7.5, leading=10, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="RCall", fontName=base_font, fontSize=9, leading=13, wordWrap="CJK", backColor=colors.HexColor("#FFF3CD"), borderPadding=5, spaceAfter=6))

    story: list[Any] = []
    for block in blocks:
        kind = block["type"]
        text = block.get("text", "").replace("<br>", "<br/>")
        if kind == "title":
            story.append(Paragraph(text, styles["RTitle"]))
        elif kind == "h1":
            story.append(Paragraph(text, styles["RH1"]))
        elif kind == "table":
            headers = block["headers"]
            data = [[Paragraph(str(h), styles["RCell"]) for h in headers]]
            for row in block["rows"]:
                data.append([Paragraph(str(c).replace("<br>", "<br/>"), styles["RCell"]) for c in row])
            if len(headers) == 4:
                widths = [26 * mm, 35 * mm, 28 * mm, 91 * mm]
            elif len(headers) == 3:
                widths = [30 * mm, 30 * mm, 110 * mm]
            else:
                widths = [35 * mm, 135 * mm]
            table = Table(data, colWidths=widths, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F5597")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
            ]))
            story.append(table)
            story.append(Spacer(1, 5))
        elif kind in ("callout", "note"):
            story.append(Paragraph(text, styles["RCall"]))
        else:
            story.append(Paragraph(text, styles["RP"]))

    pdf = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm)
    pdf.build(story)

    from pypdf import PdfReader
    return len(PdfReader(str(path)).pages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--title", default=None)
    parser.add_argument("--seed-manifest", type=Path, default=None)
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir: Path = args.run_dir

    manifest = read_json(run_dir / "manifest.json")
    report = read_json(run_dir / "run-report.json")
    strategies = read_json(run_dir / "strategies.json")
    seed = read_json(args.seed_manifest) if args.seed_manifest else read_json(run_dir / "seed-manifest.json")

    if not manifest:
        raise SystemExit(f"manifest.json not found: {run_dir / 'manifest.json'}")

    blocks = build_blocks(args, manifest, report, strategies, seed)
    md_path = out_dir / "report.md"
    docx_path = out_dir / "report.docx"
    pdf_path = out_dir / "report.pdf"
    sources_path = out_dir / "sources.json"

    md_path.write_text(render_markdown(blocks), encoding="utf-8")
    render_docx(blocks, docx_path)
    pages = render_pdf(blocks, pdf_path)
    if pages > MAX_PAGES:
        raise SystemExit(f"PDF page limit exceeded: {pages} > {MAX_PAGES}")

    source_files = [run_dir / "manifest.json", run_dir / "execution.log", run_dir / "strategies.json", run_dir / "run-report.json"]
    if args.seed_manifest:
        source_files.append(args.seed_manifest)
    elif (run_dir / "seed-manifest.json").exists():
        source_files.append(run_dir / "seed-manifest.json")

    sources_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "outputs": [str(md_path), str(docx_path), str(pdf_path)],
        "pdf_pages": pages,
        "sources": [{"path": str(p), "exists": p.exists(), "sha256": sha256(p) if p.exists() else None} for p in source_files],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"MD: {md_path} ({md_path.stat().st_size} bytes)")
    print(f"DOCX: {docx_path} ({docx_path.stat().st_size} bytes)")
    print(f"PDF: {pdf_path} ({pdf_path.stat().st_size} bytes, {pages} pages)")
    print(f"SOURCES: {sources_path}")


if __name__ == "__main__":
    main()
