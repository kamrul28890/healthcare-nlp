from __future__ import annotations

import re
from pathlib import Path


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def convert_inline(text: str) -> str:
    # Normalize markdown text that already contains escaped punctuation.
    text = re.sub(r"\\+_", "_", text)
    text = re.sub(r"\\+%", "%", text)
    text = re.sub(r"\\+#", "#", text)
    text = re.sub(r"\\+&", "&", text)
    text = re.sub(r"\\+\{", "{", text)
    text = re.sub(r"\\+\}", "}", text)

    # Preserve inline math blocks $...$ while escaping surrounding text.
    parts = re.split(r"(\$[^$]+\$)", text)
    converted: list[str] = []
    for part in parts:
        if part.startswith("$") and part.endswith("$") and len(part) >= 2:
            converted.append(part)
        else:
            converted.append(escape_latex(part))
    return "".join(converted)


def parse_table(lines: list[str], start: int) -> tuple[str, int]:
    table_lines: list[str] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        table_lines.append(lines[i].strip())
        i += 1

    # Expect header, separator, data...
    if len(table_lines) < 2:
        return convert_inline(table_lines[0]) + "\\n", i

    rows = []
    for raw in table_lines:
        cols = [c.strip() for c in raw.strip("|").split("|")]
        rows.append(cols)

    header = rows[0]
    data_rows = rows[2:] if len(rows) >= 2 else []
    ncols = len(header)
    col_spec = "l" + "c" * (ncols - 1)

    latex = []
    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    latex.append(" & ".join(convert_inline(c) for c in header) + r" \\")
    latex.append("\\midrule")
    for row in data_rows:
        row = row + [""] * (ncols - len(row))
        latex.append(" & ".join(convert_inline(c) for c in row[:ncols]) + r" \\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")
    return "\n".join(latex), i


def convert_markdown_to_latex(md_text: str) -> tuple[str, str]:
    lines = md_text.splitlines()
    body: list[str] = []
    title = "Healthcare NLP ADR Report"

    in_itemize = False
    in_enumerate = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            if in_itemize:
                body.append("\\end{itemize}")
                in_itemize = False
            if in_enumerate:
                body.append("\\end{enumerate}")
                in_enumerate = False
            body.append("")
            i += 1
            continue

        # Tables
        if stripped.startswith("|"):
            if in_itemize:
                body.append("\\end{itemize}")
                in_itemize = False
            if in_enumerate:
                body.append("\\end{enumerate}")
                in_enumerate = False
            latex_table, next_i = parse_table(lines, i)
            body.append(latex_table)
            i = next_i
            continue

        # Images
        img_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if img_match:
            if in_itemize:
                body.append("\\end{itemize}")
                in_itemize = False
            if in_enumerate:
                body.append("\\end{enumerate}")
                in_enumerate = False
            caption = convert_inline(img_match.group(1))
            path = img_match.group(2).strip()
            fig = [
                "\\begin{figure}[H]",
                "\\centering",
                f"\\includegraphics[width=0.9\\linewidth]{{{path}}}",
                f"\\caption{{{caption}}}",
                "\\end{figure}",
                "",
            ]
            body.extend(fig)
            i += 1
            continue

        # Headings
        h = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if h:
            if in_itemize:
                body.append("\\end{itemize}")
                in_itemize = False
            if in_enumerate:
                body.append("\\end{enumerate}")
                in_enumerate = False
            level = len(h.group(1))
            text = h.group(2).strip()
            text = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", text)
            if level == 1:
                title = text
                i += 1
                continue
            cmd = {2: "section", 3: "subsection", 4: "subsubsection"}.get(level, "paragraph")
            body.append(f"\\{cmd}{{{convert_inline(text)}}}")
            i += 1
            continue

        # Unordered list
        ul = re.match(r"^-\s+(.*)$", stripped)
        if ul:
            if in_enumerate:
                body.append("\\end{enumerate}")
                in_enumerate = False
            if not in_itemize:
                body.append("\\begin{itemize}")
                in_itemize = True
            body.append(f"  \\item {convert_inline(ul.group(1).strip())}")
            i += 1
            continue

        # Ordered list
        ol = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ol:
            if in_itemize:
                body.append("\\end{itemize}")
                in_itemize = False
            if not in_enumerate:
                body.append("\\begin{enumerate}")
                in_enumerate = True
            body.append(f"  \\item {convert_inline(ol.group(1).strip())}")
            i += 1
            continue

        # Paragraph
        if in_itemize:
            body.append("\\end{itemize}")
            in_itemize = False
        if in_enumerate:
            body.append("\\end{enumerate}")
            in_enumerate = False
        body.append(convert_inline(stripped))
        body.append("")
        i += 1

    if in_itemize:
        body.append("\\end{itemize}")
    if in_enumerate:
        body.append("\\end{enumerate}")

    return title, "\n".join(body)


def main() -> None:
    reports_dir = Path(__file__).resolve().parent
    md_path = reports_dir / "final_project_report.md"
    tex_path = reports_dir / "final_project_report_journal.tex"

    md_text = md_path.read_text(encoding="utf-8")
    title, latex_body = convert_markdown_to_latex(md_text)

    tex = f"""\\documentclass[11pt]{{article}}
\\usepackage[a4paper,margin=1in]{{geometry}}
\\usepackage[T1]{{fontenc}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{newtxtext,newtxmath}}
\\usepackage{{setspace}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{array}}
\\usepackage{{float}}
\\usepackage{{caption}}
\\usepackage{{hyperref}}
\\usepackage{{titlesec}}
\\usepackage{{fancyhdr}}
\\setstretch{{1.15}}
\\pagestyle{{fancy}}
\\fancyhf{{}}
\\fancyhead[L]{{Healthcare NLP ADR Project}}
\\fancyfoot[C]{{\\thepage}}
\\setlength{{\\headheight}}{{14pt}}
\\titleformat{{\\section}}{{\\large\\bfseries}}{{\\thesection.}}{{0.5em}}{{}}
\\titleformat{{\\subsection}}{{\\normalsize\\bfseries}}{{\\thesubsection.}}{{0.5em}}{{}}
\\title{{{escape_latex(title)}}}
\\author{{Kamrul Hasan}}
\\date{{April 2026}}
\\begin{{document}}
\\maketitle
\\tableofcontents
\\newpage
{latex_body}
\\end{{document}}
"""

    tex_path.write_text(tex, encoding="utf-8")
    print(f"Generated: {tex_path}")


if __name__ == "__main__":
    main()
