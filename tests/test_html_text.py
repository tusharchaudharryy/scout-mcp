"""HTML-to-text extraction: main-content detection and removal of page furniture."""

from scout_mcp.html_text import html_to_text


def test_html_to_text_keeps_structure_and_drops_boilerplate():
    title, text = html_to_text(
        "<html><head><title>Weevil &amp; palms</title><style>p{}</style></head><body>"
        "<nav>Home | About</nav><h1>Detection</h1><p>Acoustic <b>sensors</b> work.</p>"
        "<ul><li>VOC</li><li>Thermal</li></ul><script>alert('x')</script><footer>(c) site</footer></body></html>"
    )
    assert title == "Weevil & palms"
    assert text == "# Detection\n\nAcoustic sensors work.\n\n- VOC\n- Thermal"


LONG = "Acoustic sensors pick up larvae chewing inside the trunk. " * 8


def test_main_content_is_preferred_over_the_rest_of_the_page():
    _, text = html_to_text(
        "<header><a>Logo</a><ul><li>Sign in</li></ul></header>"
        "<div class='promo'>Subscribe to our newsletter</div>"
        f"<main><header><h1>Weevil detection</h1></header><p>{LONG}</p></main>"
    )
    assert text == f"# Weevil detection\n\n{LONG.strip()}"


def test_role_main_and_menus_marked_by_role_or_class():
    # the structure Sphinx docs (docs.python.org) use
    _, text = html_to_text(
        "<div class='related' role='navigation'><h3>Navigation</h3><ul><li>index<li>modules</ul></div>"
        "<div class='sphinxsidebar'><p>Table of contents</p></div>"
        f"<div class='body' role='main'><h1>ipaddress</h1><p>{LONG}</p></div>"
        "<div class='breadcrumb-trail'>Home &gt; Library</div>"
    )
    assert text.startswith("# ipaddress") and "Navigation" not in text and "Table of contents" not in text


def test_hidden_and_screen_reader_only_text_is_dropped():
    _, text = html_to_text(
        "<p>Visible.</p><span aria-hidden='true'>icon</span><div hidden>secret</div>"
        "<h2 class='sr-only'>Navigation Menu</h2><a class='px-2 js-skip-to-content'>Skip to content</a>"
        "<span class='mw-editsection'>[edit]</span><button>Toggle</button>"
    )
    assert text == "Visible."


def test_skipped_element_ends_at_its_own_closing_tag():
    _, text = html_to_text(
        "<div class='layout-sidebar'><div>a</div><div><div>b</div></div><ul><li>c<li>d</ul></div>"
        "<p>After the sidebar.</p>"
    )
    assert text == "After the sidebar."


def test_tiny_main_area_falls_back_to_the_whole_page():
    _, text = html_to_text(f"<main><h1>Title</h1></main><div><p>{LONG}</p></div>")
    assert text == f"# Title\n\n{LONG.strip()}"


def test_page_level_classes_never_hide_the_page():
    # Wikipedia puts feature flags such as "...-main-menu-disabled" on <html>; Quarto puts
    # "nav-sidebar" on <body>. Neither may be read as a menu or sidebar.
    _, text = html_to_text(
        "<html class='client-nojs vector-feature-language-in-main-menu-disabled'>"
        f"<body class='nav-sidebar floating'><main><p>{LONG}</p></main></body></html>"
    )
    assert text == LONG.strip()


def test_class_words_match_only_at_the_start_or_end_of_a_class():
    _, text = html_to_text("<div class='menu-wrapper'>Menu</div><div class='feature-menu-disabled'>Kept</div>")
    assert text == "Kept"


def test_sidebar_inside_main_content_is_kept_but_outside_is_dropped():
    _, text = html_to_text(
        f"<div class='site-sidebar'>Links</div><main><div class='layout-sidebar'>About: weevil sensors</div>"
        f"<div class='toc'>Contents</div><p>{LONG}</p></main>"
    )
    assert text.startswith("About: weevil sensors") and "Links" not in text and "Contents" not in text


def test_empty_headings_and_bullets_are_removed():
    _, text = html_to_text("<h6><img src='x.png'></h6><h5> </h5><ul><li></li></ul><h2>Real</h2><p>Text</p>")
    assert text == "## Real\n\nText"


def test_stray_and_unclosed_tags_do_not_break_extraction():
    _, text = html_to_text("</div></span><nav><ul><li>Home<li>About</nav><p>Body <b>bold</p><p>Next</p>")
    assert text == "Body bold\n\nNext"


def test_layout_wrapper_classes_cannot_hide_the_main_content():
    # Quarto (hamel.dev) wraps <main> in <div class="... page-navbar">
    _, text = html_to_text(f"<div class='quarto-container page-navbar'><main><h1>Evals</h1><p>{LONG}</p></main></div>")
    assert text.startswith("# Evals")


def test_page_without_main_falls_back_to_gentle_rules_if_strict_ones_remove_everything():
    _, text = html_to_text(f"<div class='page-navbar'><h1>Evals</h1><p>{LONG}</p></div><nav>Menu</nav>")
    assert text.startswith("# Evals") and "Menu" not in text


def test_dropdowns_inside_main_are_dropped():
    # Wikipedia's language list sits in the page header inside <main>
    _, text = html_to_text(
        "<main><header><h1>Red palm weevil</h1><div class='vector-dropdown mw-portlet-lang'>"
        f"<ul><li>Afrikaans<li>Deutsch</ul></div></header><p>{LONG}</p></main>"
    )
    assert text.startswith("# Red palm weevil\n\nAcoustic") and "Deutsch" not in text


def test_several_articles_are_treated_as_a_listing_page():
    _, text = html_to_text("<article><h2>Post one</h2></article><article><h2>Post two</h2></article>")
    assert text == "## Post one\n\n## Post two"


def test_deeply_nested_markup_does_not_overflow():
    _, text = html_to_text("<div>" * 5000 + "deep" + "</div>" * 5000)
    assert text == "deep"
