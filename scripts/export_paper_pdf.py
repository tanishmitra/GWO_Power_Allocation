"""Export a markdown project paper to a paginated PDF."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def markdown_to_plain_lines(markdown_text: str, wrap_width: int = 100) -> list[str]:
    lines: list[str] = []
    for raw in markdown_text.splitlines():
        stripped = raw.rstrip()
        if not stripped:
            lines.append("")
            continue

        if stripped.startswith("# "):
            title = stripped[2:].strip().upper()
            lines.extend([title, "=" * len(title), ""])
            continue
        if stripped.startswith("## "):
            heading = stripped[3:].strip()
            lines.extend([heading, "-" * len(heading)])
            continue
        if stripped.startswith("### "):
            heading = stripped[4:].strip()
            lines.extend([heading, ""])
            continue
        if stripped.startswith("- "):
            wrapped = textwrap.wrap(stripped[2:], width=wrap_width - 4) or [""]
            lines.append(f"- {wrapped[0]}")
            for continuation in wrapped[1:]:
                lines.append(f"  {continuation}")
            continue
        if stripped[:3] in {"1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. "}:
            prefix = stripped[:3]
            wrapped = textwrap.wrap(stripped[3:], width=wrap_width - 6) or [""]
            lines.append(f"{prefix}{wrapped[0]}")
            for continuation in wrapped[1:]:
                lines.append(f"   {continuation}")
            continue

        wrapped = textwrap.wrap(stripped, width=wrap_width) or [""]
        lines.extend(wrapped)
    return lines


def render_pdf(lines: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines_per_page = 46
    page_slices = [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)]

    with PdfPages(output_path) as pdf:
        for page_index, page_lines in enumerate(page_slices, start=1):
            fig = plt.figure(figsize=(8.27, 11.69))  # A4
            fig.patch.set_facecolor("white")
            page_text = "\n".join(page_lines)
            fig.text(
                0.08,
                0.95,
                page_text,
                va="top",
                ha="left",
                family="DejaVu Serif",
                fontsize=10.5,
                color="black",
            )
            fig.text(
                0.92,
                0.03,
                f"{page_index}",
                va="bottom",
                ha="right",
                family="DejaVu Serif",
                fontsize=9,
                color="black",
            )
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    source = Path("reports/isac_project_final_paper.md")
    output = Path("outputs/isac_project_final_paper.pdf")

    markdown_text = source.read_text(encoding="utf-8")
    lines = markdown_to_plain_lines(markdown_text)
    render_pdf(lines, output)
    print(f"Exported PDF: {output}")


if __name__ == "__main__":
    main()
