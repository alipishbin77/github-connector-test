"""HTML -> Markdown for the house webpage-to-markdown service.

Deliberately a small standard-library converter instead of a readability
port: it drops page chrome (scripts, styles, nav, footers, forms), prefers
the content of <main>/<article> when the page marks it, and maps the
structural tags an agent actually cares about. No dependency, no DOM held in
memory, output bounded by `max_chars`.
"""

import re
from html.parser import HTMLParser
from urllib.parse import urljoin

# Elements whose content is never part of the readable text.
DROP = frozenset(
    """script style noscript template svg math canvas form button select option textarea iframe object embed
       video audio nav footer aside dialog menu""".split()
)
BLOCK = frozenset(
    """p div section article main header ul ol li table tr blockquote pre dl dt dd figure figcaption address
       details summary fieldset hgroup""".split()
)
HEADINGS = {f"h{n}": n for n in range(1, 7)}
WRAPPERS = {"strong": "**", "b": "**", "em": "_", "i": "_", "del": "~~", "s": "~~"}
_SPACES = re.compile(r"\s+")


class _Markdown(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title = ""
        self.parts: list[str] = []
        self.drop = 0
        self.pre = 0
        self.quote = 0
        self.in_title = False
        self.lists: list[list] = []  # [kind, counter]
        self.links: list[tuple[int, str]] = []  # (index in parts, href)
        self.tables: list[list[int]] = []  # [rows seen, cells in current row]
        self.main: list[int | None] = [None, None]  # [start, end] index into parts

    # ------------------------------------------------------------- emitting

    def _emit(self, text: str) -> None:
        self.parts.append(text)

    def _break(self) -> None:
        self._emit("\n\n> " if self.quote else "\n\n")

    def _absolute(self, url: str | None) -> str:
        return urljoin(self.base_url, url.strip()) if url else ""

    @staticmethod
    def _attr(attrs: list[tuple[str, str | None]], name: str) -> str | None:
        return next((v for k, v in attrs if k == name), None)

    # -------------------------------------------------------------- parsing

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in DROP:
            self.drop += 1
            return
        if self.drop:
            return
        if tag == "title":
            self.in_title = True
        elif tag in ("main", "article") and self.main[0] is None:
            self.main[0] = len(self.parts)
        if tag in HEADINGS:
            self._break()
            self._emit("#" * HEADINGS[tag] + " ")
        elif tag in ("ul", "ol"):
            self.lists.append([tag, 0])
            self._break()
        elif tag == "li" and self.lists:
            kind, count = self.lists[-1]
            self.lists[-1][1] = count + 1
            self._emit("\n" + "  " * (len(self.lists) - 1) + (f"{count + 1}. " if kind == "ol" else "- "))
        elif tag == "pre":
            self.pre += 1
            self._break()
            self._emit("```\n")
        elif tag == "code" and not self.pre:
            self._emit("`")
        elif tag == "blockquote":
            self.quote += 1
            self._break()
        elif tag == "a":
            self.links.append((len(self.parts), self._absolute(self._attr(attrs, "href"))))
            self._emit("[")
        elif tag == "img":
            if src := self._absolute(self._attr(attrs, "src")):
                self._emit(f"![{(self._attr(attrs, 'alt') or '').strip()}]({src})")
        elif tag == "br":
            self._emit("\n")
        elif tag == "hr":
            self._break()
            self._emit("---")
            self._break()
        elif tag == "table":
            self.tables.append([0, 0])
            self._break()
        elif tag == "tr" and self.tables:
            self.tables[-1][1] = 0
            self._emit("\n| ")
        elif tag in ("td", "th"):
            pass
        elif tag in WRAPPERS:
            self._emit(WRAPPERS[tag])
        elif tag in BLOCK:
            self._break()

    def handle_startendtag(self, tag: str, attrs) -> None:
        if tag in ("br", "hr", "img"):
            self.handle_starttag(tag, attrs)  # void elements: no matching end tag

    def handle_endtag(self, tag: str) -> None:
        if tag in DROP:
            self.drop = max(0, self.drop - 1)
            return
        if self.drop:
            return
        if tag == "title":
            self.in_title = False
        elif tag in ("main", "article") and self.main[0] is not None and self.main[1] is None:
            self.main[1] = len(self.parts)
        if tag in HEADINGS:
            self._break()
        elif tag in ("ul", "ol"):
            if self.lists:
                self.lists.pop()
            self._break()
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
            self._emit("\n```")
            self._break()
        elif tag == "code" and not self.pre:
            self._emit("`")
        elif tag == "blockquote":
            self.quote = max(0, self.quote - 1)
            self._break()
        elif tag == "a" and self.links:
            index, href = self.links.pop()
            if href and "".join(self.parts[index + 1 :]).strip():
                self._emit(f"]({href})")
            else:  # an empty or hrefless anchor is not a link
                del self.parts[index:]
        elif tag in ("td", "th") and self.tables:
            self.tables[-1][1] += 1
            self._emit(" | ")
        elif tag == "tr" and self.tables:
            table = self.tables[-1]
            table[0] += 1
            if table[0] == 1 and table[1]:  # header separator after the first row
                self._emit("\n|" + " --- |" * table[1])
        elif tag == "table":
            if self.tables:
                self.tables.pop()
            self._break()
        elif tag in WRAPPERS:
            self._emit(WRAPPERS[tag])
        elif tag in BLOCK:
            self._break()

    def handle_data(self, data: str) -> None:
        if self.drop:
            return
        if self.in_title:
            self.title += data
            return
        if self.pre:
            self._emit(data)
            return
        text = _SPACES.sub(" ", data)
        if text == " " and (not self.parts or self.parts[-1].endswith((" ", "\n"))):
            return
        self._emit(text)

    # --------------------------------------------------------------- result

    def markdown(self) -> str:
        parts = self.parts
        start, end = self.main
        if start is not None:
            region = parts[start : end if end is not None else len(parts)]
            if len("".join(region).strip()) >= 200:  # trust <main>/<article> only if it really holds the page
                parts = region
        text = "".join(parts)
        text = "\n".join(line.rstrip() for line in text.splitlines())
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        title = _SPACES.sub(" ", self.title).strip()
        if title and not re.match(r"^#{1,6} ", text):
            text = f"# {title}\n\n{text}" if text else f"# {title}"
        return text


def html_to_markdown(html: str, *, base_url: str = "", max_chars: int = 250_000) -> str:
    """Readable content of `html` as Markdown. Relative links and images are
    resolved against `base_url`; output longer than `max_chars` is cut with a
    visible marker rather than silently truncated."""
    parser = _Markdown(base_url)
    try:
        parser.feed(html)
        parser.close()
    except AssertionError:  # html.parser gives up on some malformed markup
        pass
    text = parser.markdown()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n\n[truncated]"
    return text
