"""Macros used when building the documentation (loaded by the zensical macros extension)."""

from pathlib import Path

from pybtex.database import parse_file
from pybtex.plugin import find_plugin
from pybtex.richtext import Tag, Text
from pybtex.style import FormattedEntry
from pybtex.style.template import optional

DOCS_DIR = Path(__file__).parent

def _bold_title_style(base_style):
    """Wrap a pybtex style so that each entry is formatted as 'title (in bold) authors other infos'."""

    class BoldTitleStyle(base_style):
        # Title and authors are removed from the base template and formatted separately
        def format_title(self, e, which_field, as_sentence=True):
            if which_field == "title":
                return Text()
            return super().format_title(e, which_field, as_sentence)

        def format_btitle(self, e, which_field, as_sentence=True):
            if which_field == "title":
                return Text()
            return super().format_btitle(e, which_field, as_sentence)

        def format_names(self, role, as_sentence=True):
            if role == "author":
                return Text()
            return super().format_names(role, as_sentence)

        def format_entry(self, label, entry, bib_data=None):
            context = {"entry": entry, "style": self, "bib_data": bib_data}
            title = super().format_title(entry, "title").format_data(context)
            authors = optional[super().format_names("author")].format_data(context)
            infos = super().format_entry(label, entry, bib_data).text
            text = Text(Tag("strong", title), " ", authors, " ", infos)
            return FormattedEntry(entry.key, text, label)

    return BoldTitleStyle

def define_env(env):
    @env.macro
    def bibliography(bib_file="bibliography.bib", style="plain"):
        """Render every entry of a .bib file (relative to the docs folder) as a markdown list.
        Each entry is displayed on a single line, with the title in bold."""
        bib = parse_file(DOCS_DIR / bib_file)
        # Zotero exports some urls in the 'howpublished' field, which is not rendered as a hyperlink
        for entry in bib.entries.values():
            howpublished = entry.fields.get("howpublished", "")
            if howpublished.startswith("http") and "url" not in entry.fields and "doi" not in entry.fields:
                entry.fields["url"] = entry.fields.pop("howpublished")
                entry.fields.pop("urldate", None)
        style_cls = _bold_title_style(find_plugin("pybtex.style.formatting", style))
        formatted = style_cls().format_bibliography(bib)
        return "\n".join(
            "- " + " ".join(entry.text.render_as("markdown").split("\n"))
            for entry in formatted
        )
