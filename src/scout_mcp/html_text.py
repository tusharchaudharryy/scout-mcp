"""Turn an HTML page into readable text with light Markdown structure.

The page is first parsed into an element tree. If it marks a main content area (`<main>`,
`role="main"`, or a single `<article>`), text is taken from that area only. Otherwise the whole
page is used. Either way, page furniture is dropped: menus, sidebars, footers, hidden and
screen-reader-only text, scripts and forms.

Because the main area is found before any furniture rules run, classes on the layout wrappers
around it (for example `page-navbar` or `main-menu-disabled`) cannot hide the content.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from html.parser import HTMLParser

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
_SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "iframe", "nav", "footer", "aside", "form",
              "head", "button", "select", "dialog"}
_SKIP_ROLES = {"navigation", "banner", "contentinfo", "complementary", "search", "menu", "menubar", "dialog"}
# Class/id words that mark furniture. A word must start or end a class/id ("layout-sidebar",
# "breadcrumb-item"), so long names that merely contain it ("feature-main-menu-disabled") do not match.
# Inside the main area only the narrower set applies: a sidebar there is usually part of the content,
# such as a repository's "About" box.
_SKIP_WORDS = {"sidebar", "breadcrumb", "breadcrumbs", "cookie", "cookies", "navbar", "navbox", "menu", "toc",
               "dropdown", "editsection"}
_SKIP_WORDS_IN_MAIN = {"breadcrumb", "breadcrumbs", "cookie", "cookies", "navbox", "toc", "dropdown", "editsection"}
_SKIP_CLASSES = {"sr-only", "visually-hidden", "screen-reader-text", "noprint", "mw-jump-link", "skip-link",
                 "skip-to-content", "js-skip-to-content"}
_NEVER_SKIP = {"html", "body", "main", "article"}
_BLOCK = {"p", "div", "section", "article", "main", "header", "blockquote", "pre", "table", "tr", "ul", "ol",
          "dl", "dt", "dd", "figure", "figcaption"}
_HEADING = re.compile(r"h[1-6]")


class _Node:
    __slots__ = ("tag", "attrs", "children")

    def __init__(self, tag: str, attrs: dict[str, str | None]) -> None:
        self.tag = tag
        self.attrs = attrs
        self.children: list[_Node | str] = []

    @property
    def role(self) -> str:
        return (self.attrs.get("role") or "").lower()


class _TreeBuilder(HTMLParser):
    """Builds a forgiving element tree: a closing tag closes everything opened after its element
    (so unclosed `<p>`/`<li>` are handled), and stray closing tags are ignored."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("#root", {})
        self._stack = [self.root]
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title" and not any(n.tag == "svg" for n in self._stack):
            self._in_title = True
            return
        node = _Node(tag, dict(attrs))
        self._stack[-1].children.append(node)
        if tag not in _VOID:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._stack[-1].children.append(_Node(tag, dict(attrs)))

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._in_title:
            self._in_title = False
            return
        for i in range(len(self._stack) - 1, 0, -1):
            if self._stack[i].tag == tag:
                del self._stack[i:]
                return

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        else:
            self._stack[-1].children.append(data)


def _is_main(node: _Node) -> bool:
    return node.tag in ("main", "article") or node.role == "main"


def _is_furniture(node: _Node, *, in_main: bool, class_rules: bool) -> bool:
    """True for menus, sidebars, hidden text and other parts of a page that are not its content."""
    if node.tag in _NEVER_SKIP or node.role == "main":
        return False
    if node.tag in _SKIP_TAGS or (node.tag == "header" and not in_main):
        return True
    if node.role in _SKIP_ROLES:
        return True
    if (node.attrs.get("aria-hidden") or "").lower() == "true" or "hidden" in node.attrs:
        return True
    words = (_SKIP_WORDS_IN_MAIN if in_main else _SKIP_WORDS) if class_rules else set()
    for token in f"{node.attrs.get('class') or ''} {node.attrs.get('id') or ''}".lower().split():
        parts = re.split(r"[-_]", token)
        if token in _SKIP_CLASSES or parts[0] in words or parts[-1] in words:
            return True
    return False


def tidy(text: str) -> str:
    """Collapse whitespace, drop markup left with no text, and keep at most one blank line."""
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n")]
    lines = ["" if re.fullmatch(r"[#\-|\s]+", line) else line for line in lines]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _render(start: _Node, *, in_main: bool, class_rules: bool) -> str:
    # Iterative walk: badly nested pages can be thousands of levels deep.
    out: list[str] = []
    stack: list[tuple[_Node | str, bool, bool]] = [(start, in_main, False)]
    while stack:
        item, main, closing = stack.pop()
        if isinstance(item, str):
            out.append(item)
            continue
        tag = item.tag
        if closing:
            if tag in _BLOCK or _HEADING.fullmatch(tag):
                out.append("\n\n")
            continue
        if item is not start and _is_furniture(item, in_main=main, class_rules=class_rules):
            continue
        main = main or _is_main(item)
        if tag in _BLOCK:
            out.append("\n\n")
        elif _HEADING.fullmatch(tag):
            out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            out.append("\n- ")
        elif tag in ("br", "hr"):
            out.append("\n")
        elif tag in ("td", "th"):
            out.append(" | ")
        stack.append((item, main, True))
        stack.extend((child, main, False) for child in reversed(item.children))
    return tidy("".join(out))


def _find(root: _Node, match: Callable[[_Node], bool]) -> list[_Node]:
    found: list[_Node] = []
    stack: list[_Node | str] = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, _Node):
            if match(node):
                found.append(node)
            stack.extend(reversed(node.children))
    return found


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, text) for an HTML document, preferring its main content area."""
    builder = _TreeBuilder()
    builder.feed(html)
    builder.close()
    title = tidy(" ".join(builder.title_parts))

    page = _render(builder.root, in_main=False, class_rules=True)
    lenient = _render(builder.root, in_main=False, class_rules=False)
    if len(page) < 0.2 * len(lenient):  # a layout wrapper was mistaken for furniture
        page = lenient

    mains = _find(builder.root, lambda n: n.tag == "main" or n.role == "main")
    if not mains:
        articles = _find(builder.root, lambda n: n.tag == "article")
        mains = articles if len(articles) == 1 else []  # several articles = a listing page
    best = max((_render(m, in_main=True, class_rules=True) for m in mains), key=len, default="")
    # Use the main area unless it is a small fragment of the page (e.g. only a heading).
    if best and (len(best) >= 2000 or len(best) >= 0.25 * len(page)):
        return title, best
    return title, page
