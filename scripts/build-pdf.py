#!/usr/bin/env python3
"""Build the whole book as one PDF.

    python3 scripts/build-pdf.py                 # HTML + PDF
    python3 scripts/build-pdf.py --html-only     # just the HTML, to print yourself
    python3 scripts/build-pdf.py --no-myst       # reuse the existing _build/site content

The source is **MyST's own parse output** — the JSON under ``_build/site/content/`` that
``myst build --site`` writes — rather than the markdown. That matters: it means the PDF cannot
disagree with the website about what a chapter says, because both render the same tree. Directives
are already resolved in it, so every ``{literalinclude}`` carries the real code and every
``{include}`` carries the real generated table, with no second implementation of either to drift.

Why not ``myst build --pdf``? It needs a LaTeX or Typst template fetched from GitHub plus a TeX
toolchain, and in a network-restricted environment neither is available. This path needs only
Chromium, which is also what produces the PDF a reader gets by pressing Print on the HTML.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "_build" / "site" / "content"
OUT_DIR = ROOT / "_build" / "exports"
STEM = "llm-serving-from-scratch"

#: Chromium locations to try, in order. The second is where this project's container keeps the
#: browser Playwright manages; the rest are ordinary system installs.
CHROMIUM_CANDIDATES = (
    "chromium",
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
)


class UnknownNodeError(Exception):
    """A node type the renderer does not handle.

    Raised rather than skipped, deliberately. A renderer that ignores what it does not recognise
    silently drops content from the PDF, and the only symptom is a missing paragraph nobody notices
    — which is exactly the class of failure this book spends thirty chapters complaining about. If
    a chapter starts using a new directive, this fails loudly and gets one more ``elif``.
    """


# -- the table of contents ---------------------------------------------------------------


@dataclass
class Page:
    slug: str
    source: str
    mdast: dict
    title: str = ""


@dataclass
class Part:
    title: str
    pages: list[Page] = field(default_factory=list)


def load_pages() -> dict[str, Page]:
    """Every parsed page, indexed by its repo-relative source path.

    Indexed by ``location`` rather than by a slug derived from the filename: MyST's slugging rules
    are its business, and guessing them would break the day they change.
    """
    if not CONTENT.is_dir():
        raise SystemExit(
            f"{CONTENT} does not exist — run `myst build --site` first, or drop --no-myst"
        )
    pages = {}
    for path in sorted(CONTENT.glob("*.json")):
        data = json.loads(path.read_text())
        location = (data.get("location") or "").lstrip("/")
        if not location:
            continue
        pages[location] = Page(slug=data["slug"], source=location, mdast=data["mdast"])
    if not pages:
        raise SystemExit(f"no parsed pages found in {CONTENT}")
    return pages


def load_structure(pages: dict[str, Page]) -> tuple[Page | None, list[Part]]:
    """Read myst.yml's table of contents, so the PDF's order is the website's order.

    Returns the front-matter page (index.md, which becomes the cover's blurb) separately from the
    parts, because it is the one entry that is not part of a numbered sequence.
    """
    config = yaml.safe_load((ROOT / "myst.yml").read_text())
    toc = config.get("project", {}).get("toc") or []

    front: Page | None = None
    parts: list[Part] = []
    for entry in toc:
        if "file" in entry and not parts:
            front = pages.get(entry["file"].lstrip("./"))
            continue
        if "title" not in entry:
            continue
        part = Part(title=entry["title"])
        for child in entry.get("children") or []:
            source = child.get("file")
            if not source:
                continue
            page = pages.get(source.lstrip("./"))
            if page is None:
                raise SystemExit(f"{source} is in myst.yml but was not parsed — rebuild the site")
            part.pages.append(page)
        if part.pages:
            parts.append(part)

    if not parts:
        raise SystemExit("myst.yml's toc has no parts with files in it")
    return front, parts


# -- rendering ---------------------------------------------------------------------------


def anchor(slug: str, html_id: str) -> str:
    """A document-unique id.

    Every page has its own ``the-cost`` and ``key-takeaways``, so the per-page ids collide dozens of
    times once the book is one document. Namespacing by slug is what makes the internal links in the
    PDF land on the right chapter instead of on chapter one.
    """
    return f"{slug}--{html_id}"


class Renderer:
    """mdast to HTML, for the node types this book actually uses."""

    def __init__(self, slug: str) -> None:
        self.slug = slug
        #: headings encountered, for the table of contents
        self.headings: list[tuple[int, str, str]] = []

    def children(self, node: dict) -> str:
        return "".join(self.render(child) for child in node.get("children") or [])

    def render(self, node: dict) -> str:
        kind = node.get("type")

        if kind == "text":
            return html.escape(node.get("value", ""))
        if kind == "inlineCode":
            return f"<code>{html.escape(node.get('value', ''))}</code>"
        if kind == "strong":
            return f"<strong>{self.children(node)}</strong>"
        if kind == "emphasis":
            return f"<em>{self.children(node)}</em>"
        if kind == "paragraph":
            return f"<p>{self.children(node)}</p>"
        if kind == "blockquote":
            return f"<blockquote>{self.children(node)}</blockquote>"
        if kind == "list":
            tag = "ol" if node.get("ordered") else "ul"
            return f"<{tag}>{self.children(node)}</{tag}>"
        if kind == "listItem":
            return f"<li>{self.children(node)}</li>"
        if kind == "link":
            url = html.escape(node.get("url", ""), quote=True)
            return f'<a href="{url}">{self.children(node)}</a>'

        if kind == "heading":
            return self.heading(node)
        if kind == "crossReference":
            return self.cross_reference(node)
        if kind == "code":
            return self.code(node)
        if kind == "table":
            return self.table(node)
        if kind == "admonition":
            return self.admonition(node)

        # Structural wrappers: the interesting content is underneath.
        if kind in {"root", "block", "include", "container"}:
            return self.children(node)
        # A comment is the "do not edit by hand" banner on a generated fragment.
        if kind == "comment":
            return ""

        raise UnknownNodeError(f"{kind!r} in {self.slug} — add a branch for it in Renderer.render")

    def heading(self, node: dict) -> str:
        depth = min(max(int(node.get("depth", 2)), 2), 6)
        html_id = node.get("html_id") or node.get("identifier") or ""
        text = self.children(node)
        ident = anchor(self.slug, html_id) if html_id else ""
        self.headings.append((depth, text, ident))
        attr = f' id="{html.escape(ident, quote=True)}"' if ident else ""
        return f"<h{depth}{attr}>{text}</h{depth}>"

    def cross_reference(self, node: dict) -> str:
        """An internal reference, retargeted at this document rather than at the website.

        A cross-reference to another page carries the target page's URL; one to a heading in the
        same page carries no URL at all, so the current slug is the right default.
        """
        target_slug = (node.get("url") or "").strip("/") or self.slug
        html_id = node.get("html_id") or node.get("identifier")
        label = self.children(node) or html.escape(str(html_id or ""))
        if not html_id:
            return label
        return f'<a class="xref" href="#{html.escape(anchor(target_slug, html_id), quote=True)}">{label}</a>'

    def code(self, node: dict) -> str:
        body = html.escape(node.get("value", ""))
        lang = html.escape(str(node.get("lang") or ""), quote=True)
        filename = node.get("filename")
        caption = f"<figcaption>{html.escape(str(filename))}</figcaption>" if filename else ""
        return (
            f'<figure class="code">{caption}'
            f'<pre><code class="language-{lang}">{body}</code></pre></figure>'
        )

    def table(self, node: dict) -> str:
        head, body = [], []
        for row in node.get("children") or []:
            cells = row.get("children") or []
            is_header = bool(cells) and all(c.get("header") for c in cells)
            # A header row with nothing in it comes from the borderless two-column grids the
            # chapter headers use. Rendering it leaves an empty band that reads as a bug.
            if is_header and not any(cell.get("children") for cell in cells):
                continue
            rendered = "".join(
                f"<{'th' if cell.get('header') else 'td'}>{self.children(cell)}"
                f"</{'th' if cell.get('header') else 'td'}>"
                for cell in cells
            )
            (head if is_header and not body else body).append(f"<tr>{rendered}</tr>")
        parts = []
        if head:
            parts.append(f"<thead>{''.join(head)}</thead>")
        if body:
            parts.append(f"<tbody>{''.join(body)}</tbody>")
        return f'<div class="table-wrap"><table>{"".join(parts)}</table></div>'

    def admonition(self, node: dict) -> str:
        kind = html.escape(str(node.get("kind") or "note"), quote=True)
        title, rest = "", []
        for child in node.get("children") or []:
            if child.get("type") == "admonitionTitle":
                title = self.children(child)
            else:
                rest.append(self.render(child))
        heading = f'<p class="admonition-title">{title}</p>' if title else ""
        return f'<aside class="admonition {kind}">{heading}{"".join(rest)}</aside>'


# -- the document ------------------------------------------------------------------------


def stylesheet() -> str:
    """Print CSS. Every rule here exists because something looked wrong without it."""
    return """
@page { size: A4; margin: 20mm 18mm 22mm; }

html { font-size: 10.5pt; }
body {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  line-height: 1.5;
  color: #1a1a1a;
}
h1, h2, h3, h4, h5, h6 { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; line-height: 1.25; }

/* Each chapter and each part divider starts a fresh sheet. */
section.page, section.part { break-before: page; }
section.cover { break-after: page; }
section.contents { break-after: page; }

/* A heading stranded at the foot of a page is the most common print-CSS failure. */
h2, h3, h4, h5 { break-after: avoid; page-break-after: avoid; }
h2 { font-size: 1.55rem; margin: 0 0 1rem; }
/* A chapter's own title opens its page; anything above it is wasted paper. */
section.page > h2:first-child { margin-top: 0; }
h3 { font-size: 1.15rem; margin: 1.6rem 0 0.4rem; }
h4 { font-size: 1rem; margin: 1.2rem 0 0.3rem; }
p, li { orphans: 3; widows: 3; }

/* Cover. The title block sits on the upper third rather than dead centre — centred looks
   accidental on a page this tall, and leaves the blurb stranded. */
section.cover { display: flex; flex-direction: column; min-height: 232mm; }
.cover .titleblock { margin-top: 22%; }
.cover h1 { font-size: 3rem; margin: 0 0 0.5rem; letter-spacing: -0.02em; line-height: 1.1; }
.cover .tagline { font-size: 1.25rem; font-style: italic; color: #444; margin: 0; }
.cover .rule { border: 0; border-top: 1.5pt solid #1a1a1a; width: 5rem; margin: 1.6rem 0; }
.cover .blurb { font-size: 1rem; color: #333; max-width: 32em; margin: 0; }
.cover .meta { margin-top: auto; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
               font-size: 0.8rem; color: #555; }
.cover .meta code { font-size: 0.78rem; }

/* Contents */
.contents h1 { font-size: 1.8rem; margin: 0 0 1.2rem; }
.contents ol { list-style: none; margin: 0; padding: 0; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
.contents .part-row { margin: 1.1rem 0 0.35rem; font-weight: 700; font-size: 0.95rem; }
.contents .chapter-row { margin: 0.12rem 0 0.12rem 1.2rem; font-size: 0.9rem; }
.contents a { color: #1a1a1a; text-decoration: none; }
.contents .note { font-size: 0.8rem; color: #666; margin-top: 1.6rem; font-style: italic; }

/* Part dividers */
section.part { display: flex; flex-direction: column; justify-content: center; min-height: 200mm; }
.part h1 { font-size: 2.2rem; margin: 0; }
.part .part-chapters { margin-top: 1.2rem; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                       font-size: 0.9rem; color: #555; }

/* Code. Wrapping matters more than fidelity here: a clipped line in print is lost, and the
   longest lines in this book are ~100 characters of Python. */
figure.code { margin: 0.9rem 0; break-inside: avoid; }
figure.code figcaption {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 0.72rem;
  color: #555; padding: 0 0 0.25rem 0.1rem;
}
pre {
  margin: 0; padding: 0.6rem 0.7rem; background: #f6f7f9; border: 0.5pt solid #dcdfe4;
  border-radius: 2pt; font-size: 0.72rem; line-height: 1.38;
  white-space: pre-wrap; overflow-wrap: break-word;
}
code { font-family: "SFMono-Regular", Menlo, Consolas, monospace; }
p code, li code, td code, th code { font-size: 0.82em; background: #f2f3f5; padding: 0 2pt; border-radius: 2pt; }

/* Tables. The widest in the book is seven columns of numbers; small type and fixed layout keep
   it on the page instead of off the right edge. */
.table-wrap { margin: 0.9rem 0; break-inside: avoid; }
table { width: 100%; border-collapse: collapse; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        font-size: 0.72rem; table-layout: auto; }
th, td { border: 0.5pt solid #d6d9de; padding: 3pt 5pt; text-align: left; vertical-align: top;
         overflow-wrap: break-word; }
thead th { background: #eef0f3; font-weight: 700; }

/* Admonitions */
aside.admonition {
  margin: 1rem 0; padding: 0.6rem 0.8rem; border-left: 2.5pt solid #8a8f98; background: #f7f8fa;
  break-inside: avoid;
}
aside.admonition.warning { border-left-color: #c2410c; background: #fdf6f2; }
aside.admonition.note { border-left-color: #1d4ed8; background: #f3f6fd; }
aside.admonition .admonition-title {
  margin: 0 0 0.35rem; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-weight: 700; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;
}
aside.admonition > :last-child { margin-bottom: 0; }

blockquote {
  margin: 1rem 0; padding: 0.1rem 0 0.1rem 0.9rem; border-left: 2pt solid #c9ced6;
  color: #333; font-style: italic;
}

a { color: #1d4ed8; }
a.xref { text-decoration: none; font-weight: 600; }
"""


def contents_html(front: Page | None, parts: list[Part]) -> str:
    """A linked table of contents.

    No page numbers, and that is a limitation rather than an oversight: the page a chapter lands on
    is not known until the browser paginates, which is after this HTML is written. The entries are
    hyperlinks, which is what a reader of a PDF actually uses.
    """
    rows = []
    if front is not None:
        rows.append(
            f'<li class="chapter-row"><a href="#{front.slug}">{html.escape(front.title)}</a></li>'
        )
    for index, part in enumerate(parts, start=1):
        rows.append(
            f'<li class="part-row"><a href="#part-{index}">{html.escape(part.title)}</a></li>'
        )
        for page in part.pages:
            rows.append(
                f'<li class="chapter-row"><a href="#{page.slug}">{html.escape(page.title)}</a></li>'
            )
    return (
        '<section class="contents"><h1>Contents</h1><ol>'
        + "".join(rows)
        + "</ol>"
        + '<p class="note">Entries are links. Page numbers are omitted because pagination happens '
        "after this document is assembled.</p></section>"
    )


def build_html(front: Page | None, parts: list[Part], title: str, tagline: str, blurb: str) -> str:
    """Assemble the single document, rendering every page in the website's order."""
    body: list[str] = []

    commit = git_describe()
    built = datetime.now(UTC).strftime("%d %B %Y")
    body.append(
        '<section class="cover"><div class="titleblock">'
        f"<h1>{html.escape(title)}</h1>"
        f'<p class="tagline">{html.escape(tagline)}</p>'
        '<hr class="rule">'
        f'<p class="blurb">{html.escape(blurb)}</p></div>'
        f'<p class="meta">Chris Snow · built {html.escape(built)}'
        + (f" · <code>{html.escape(commit)}</code>" if commit else "")
        + "<br>Prose CC-BY-NC-4.0 · engine code Apache-2.0</p></section>"
    )

    def render_page(page: Page, tag: str = "page") -> str:
        renderer = Renderer(page.slug)
        inner = renderer.render(page.mdast)
        page.title = next((text for depth, text, _ in renderer.headings if depth == 2), page.slug)
        # The rendered title may contain markup; the contents list wants plain text.
        page.title = strip_tags(page.title)
        return f'<section class="{tag}" id="{html.escape(page.slug, quote=True)}">{inner}</section>'

    rendered_front = render_page(front, tag="page") if front is not None else ""
    rendered_parts = []
    for index, part in enumerate(parts, start=1):
        pages_html = [render_page(page) for page in part.pages]
        listing = " · ".join(html.escape(p.title) for p in part.pages)
        rendered_parts.append(
            f'<section class="part" id="part-{index}"><h1>{html.escape(part.title)}</h1>'
            f'<p class="part-chapters">{listing}</p></section>' + "".join(pages_html)
        )

    body.append(contents_html(front, parts))
    if rendered_front:
        body.append(rendered_front)
    body.extend(rendered_parts)

    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{stylesheet()}</style></head>"
        f"<body>{''.join(body)}</body></html>\n"
    )


def strip_tags(text: str) -> str:
    out, depth = [], 0
    for char in text:
        if char == "<":
            depth += 1
        elif char == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(char)
    return html.unescape("".join(out)).strip()


def git_describe() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


# -- turning it into a PDF ---------------------------------------------------------------


def find_chromium() -> str | None:
    for candidate in CHROMIUM_CANDIDATES:
        found = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if found:
            return found
    return None


def to_pdf(html_path: Path, pdf_path: Path, title: str) -> bool:
    """Render the HTML to PDF, preferring Playwright for its page numbers.

    Playwright can supply a footer template, so the PDF gets "page N of M"; the Chromium CLI cannot,
    and its built-in footer prints the source file's URL, which is worse than nothing. Both paths use
    the same browser and the same CSS, so the only difference is the footer.
    """
    url = html_path.as_uri()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return _to_pdf_cli(url, pdf_path)

    footer = (
        '<div style="width:100%;font:8pt Helvetica,Arial,sans-serif;color:#666;'
        'padding:0 18mm;display:flex;justify-content:space-between">'
        f"<span>{html.escape(title)}</span>"
        '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>'
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page()
        page.goto(url, wait_until="load")
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=footer,
            margin={"top": "20mm", "bottom": "22mm", "left": "18mm", "right": "18mm"},
        )
        browser.close()
    return True


def _to_pdf_cli(url: str, pdf_path: Path) -> bool:
    binary = find_chromium()
    if not binary:
        return False
    subprocess.run(
        [
            binary,
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            url,
        ],
        check=True,
        capture_output=True,
    )
    return pdf_path.exists()


def page_count(pdf_path: Path) -> int:
    """Count pages without a PDF library, so this script keeps its single dependency."""
    data = pdf_path.read_bytes()
    return max(
        data.count(b"/Type /Page\n"), data.count(b"/Type/Page\n"), data.count(b"/Type /Page ")
    )


def reparse() -> bool:
    """Re-parse the sources, and prove it happened.

    ``myst build`` (bare) writes the content JSON this script reads. It is deliberately *not*
    ``--site`` or ``--html``: those fetch the website theme first, so in an environment that cannot
    reach the template registry they abort having parsed nothing — and the exit code alone cannot
    tell that apart from a content error. ``scripts/ci-check.sh`` has the same problem and solves it
    the same way: require the page count in the output, which only appears once parsing has actually
    finished. Whether every page the table of contents names came through is checked separately, by
    :func:`load_structure`.
    """
    print("== myst build ==")
    result = subprocess.run(["myst", "build"], cwd=ROOT, capture_output=True, text=True)
    output = result.stdout + result.stderr
    match = re.search(r"Built (\d+) pages", output)
    if not match:
        print(output.strip()[-2000:], file=sys.stderr)
        print("\nmyst parsed no pages; not assembling a PDF from stale content.", file=sys.stderr)
        return False
    print(f"  parsed {match.group(1)} pages")
    if result.returncode != 0:
        # Reached when the parse succeeded and only the theme download failed. Say so rather than
        # failing: the theme decides how the website looks and has nothing to do with this PDF.
        print("  (myst exited non-zero after parsing — usually the site theme, not the content)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / f"{STEM}.pdf")
    parser.add_argument("--html-only", action="store_true", help="write the HTML and stop")
    parser.add_argument(
        "--no-myst", action="store_true", help="reuse the existing _build/site content"
    )
    args = parser.parse_args()

    if not args.no_myst and not reparse():
        return 1

    pages = load_pages()
    front, parts = load_structure(pages)

    config = yaml.safe_load((ROOT / "myst.yml").read_text()).get("project", {})
    title = config.get("title", "LLM Serving from Scratch")
    blurb = " ".join((config.get("description") or "").split())

    document = build_html(
        front,
        parts,
        title=title,
        tagline="Build a production inference engine, one measurement at a time.",
        blurb=blurb,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    html_path = args.out.with_suffix(".html")
    html_path.write_text(document)
    chapters = sum(len(part.pages) for part in parts)
    print(f"  wrote {html_path.relative_to(ROOT)} ({chapters} chapters in {len(parts)} parts)")

    if args.html_only:
        print("  --html-only: open it in a browser and print to PDF.")
        return 0

    if not to_pdf(html_path, args.out, title):
        print(
            "\nNo Chromium found, so only the HTML was written.\n"
            "  Open it and print to PDF, or install a browser:\n"
            "    pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        return 1

    size_mb = args.out.stat().st_size / 1e6
    print(f"  wrote {args.out.relative_to(ROOT)} ({page_count(args.out)} pages, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
