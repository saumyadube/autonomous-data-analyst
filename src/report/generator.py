"""
Layer 5: Report Generator
---------------------------
Generates a downloadable PDF report from
the analysis results and LLM narrative.
"""

from fpdf import FPDF
import base64
import io
from datetime import datetime
import tempfile
import os


class AnalysisReport(FPDF):
    """Custom PDF class with header and footer."""

    def __init__(self, dataset_name: str):
        super().__init__()
        self.dataset_name = dataset_name
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(31, 56, 100)
        self.cell(0, 8, "LLM-Powered Autonomous Data Analyst — Saumaya Dube", align="L")
        self.ln(4)
        self.set_draw_color(31, 56, 100)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Rama University", align="C")

    def chapter_title(self, title: str):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(31, 56, 100)
        self.set_fill_color(214, 228, 240)
        self.cell(0, 10, title, fill=True, ln=True)
        self.ln(3)

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(46, 80, 144)
        self.cell(0, 8, title, ln=True)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_key_value(self, key: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(31, 56, 100)
        self.cell(60, 6, key + ":", ln=False)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.cell(0, 6, str(value), ln=True)


def generate_pdf_report(
    dataset_name: str,
    profile: dict,
    narrative: str,
    tool_summaries: list
) -> bytes:
    """
    Generate a complete PDF report.

    Args:
        dataset_name: Name of the uploaded CSV file
        profile: Dataset profile dict from profiler.py
        narrative: LLM-generated narrative string
        tool_summaries: List of tool result summaries

    Returns:
        PDF as bytes
    """
    pdf = AnalysisReport(dataset_name)

    # ── TITLE PAGE ────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(31, 56, 100)
    pdf.ln(20)
    pdf.cell(0, 12, "Autonomous Data Analysis Report", align="C", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(70, 114, 196)
    pdf.cell(0, 10, f"Dataset: {dataset_name}", align="C", ln=True)
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", align="C", ln=True)
    pdf.cell(0, 8, "Submitted by: Saumaya Dube | Rama University, Kanpur", align="C", ln=True)
    pdf.cell(0, 8, "Project: LLM-Powered Autonomous Data Analyst", align="C", ln=True)

    # Decorative line
    pdf.ln(10)
    pdf.set_draw_color(31, 56, 100)
    pdf.set_line_width(1)
    pdf.line(30, pdf.get_y(), 180, pdf.get_y())

    # ── DATASET OVERVIEW ─────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("1. Dataset Overview")
    pdf.add_key_value("Dataset Name",    dataset_name)
    pdf.add_key_value("Rows",            str(profile["shape"]["rows"]))
    pdf.add_key_value("Columns",         str(profile["shape"]["columns"]))
    pdf.add_key_value("Total Missing",   str(profile["missing_total"]))
    pdf.add_key_value("Duplicate Rows",  str(profile["duplicate_rows"]))
    pdf.add_key_value("Analysis Date",   datetime.now().strftime("%Y-%m-%d"))
    pdf.ln(4)

    pdf.section_title("Column Summary")
    for col in profile["columns"]:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(31, 56, 100)
        pdf.cell(60, 5, col["name"], ln=False)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5,
            f"Type: {col['dtype']} | Nulls: {col['null_pct']}% | Unique: {col['unique_count']}",
            ln=True)

    # ── ANALYTICAL REPORT (LLM narrative) ────────────────
    pdf.add_page()
    pdf.chapter_title("2. Analytical Report")

    # Split narrative by sections and format
    lines = narrative.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(2)
            continue
        # Section headers (##)
        if line.startswith("## "):
            pdf.ln(3)
            pdf.section_title(line.replace("## ", "").replace("#", "").strip())
        elif line.startswith("# "):
            pdf.chapter_title(line.replace("# ", "").strip())
        # Bullet points
        elif line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            bullet_text = "  • " + line.lstrip("-* •").strip()
            pdf.multi_cell(0, 6, bullet_text)
        # Bold text (remove markdown)
        elif "**" in line:
            clean_line = line.replace("**", "")
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 6, clean_line)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 6, line)

    # ── TOOL RESULTS SUMMARY ─────────────────────────────
    pdf.add_page()
    pdf.chapter_title("3. Detailed Tool Results")

    for i, summary in enumerate(tool_summaries, 1):
        if not summary:
            continue
        first_line = summary.split("\n")[0]
        pdf.section_title(f"Tool {i}: {first_line[:60]}")
        rest = "\n".join(summary.split("\n")[1:])
        pdf.body_text(rest[:1500])  # Limit length per tool
        pdf.ln(3)

    # ── METHODOLOGY ──────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("4. Methodology")
    pdf.body_text(
        "This report was generated by the LLM-Powered Autonomous Data Analyst system, "
        "a capstone project developed by Saumaya Dube at Rama University, Kanpur. "
        "The system uses a five-layer agentic architecture:\n\n"
        "Layer 1 - Data Ingestion & Profiling: Reads and profiles the uploaded CSV file.\n"
        "Layer 2 - LLM Orchestration: Uses LLaMA 3 70B (via Groq) to plan the analysis.\n"
        "Layer 3 - Tool Execution: Runs statistical tools (SciPy, Scikit-learn, Pandas).\n"
        "Layer 4 - Insight Synthesis: LLM synthesizes tool outputs into business insights.\n"
        "Layer 5 - Report Generation: Assembles findings into this PDF report.\n\n"
        "The analysis is grounded in actual computational results from the dataset, "
        "not in the model's training knowledge, ensuring factual accuracy."
    )

    return bytes(pdf.output())
