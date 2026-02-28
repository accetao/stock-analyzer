"""Systematic URL routing rewrite using exact string boundaries."""

with open(r"c:\3funspace\stock-analyzer\app.py", "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# ═══ 1. Replace everything from _read_url_params to go_to_analysis ═══
start_marker = "def _read_url_params():"
end_marker = "\ndef go_to_analysis"

start_idx = content.index(start_marker)
end_idx = content.index(end_marker)

new_funcs = '''def _read_url_params():
    """Detect browser back/forward or first load from URL params.

    Compares URL slug to _url_expected (what we last wrote or acknowledged).
    - Match    -> URL is what we expect, sidebar owns navigation, skip.
    - Mismatch -> URL changed externally (browser back/forward or first load),
                  update session state to follow the URL.
    """
    if "nav_to" in st.session_state:
        return
    qp = st.query_params
    slug = qp.get("page", "")
    if not slug or slug not in _SLUG_TO_PAGE:
        return
    # If URL matches what we expect, nothing changed externally
    if slug == st.session_state.get("_url_expected", ""):
        return
    # URL changed externally -> follow it
    url_page = _SLUG_TO_PAGE[slug]
    st.session_state["main_nav"] = url_page
    st.session_state["_url_expected"] = slug  # acknowledge the new URL
    # Handle analysis symbol
    sym = qp.get("symbol", "")
    if slug == "analysis" and sym:
        st.session_state["analyze_symbol"] = sym.upper()
        st.session_state["_analysis_active"] = True
        st.session_state.pop("sym_main", None)
        st.session_state["_sym_main_result"] = sym.upper()
    elif slug != "analysis":
        st.session_state.pop("_analysis_active", None)


def _write_url_params(page_label: str, symbol: str = ""):
    """Write the current page to the browser URL bar (once per render).

    Called at the very end of the script. Updates _url_expected so
    _read_url_params can distinguish our writes from browser back/forward.
    Only calls from_dict when URL actually needs changing.
    """
    slug = _PAGE_TO_SLUG.get(page_label, "dashboard")
    want = {"page": slug}
    if symbol:
        want["symbol"] = symbol.upper()
    # Update our expectation BEFORE writing
    st.session_state["_url_expected"] = slug
    # Read current URL
    qp = st.query_params
    current = {}
    for k in ("page", "symbol"):
        v = qp.get(k, "")
        if v:
            current[k] = v
    # Only write when URL differs (from_dict triggers rerun)
    if current != want:
        st.query_params.from_dict(want)
'''

content = content[:start_idx] + new_funcs + content[end_idx:]
changes += 1
print("1. Replaced _read/_write URL functions")

# ═══ 2. Simplify bottom URL sync block ═══
old_bottom_line = 'if not st.session_state.pop("_url_nav_from_browser", False):'
if old_bottom_line in content:
    # Remove the if guard and dedent the block
    content = content.replace(
        '# Skip if this render was triggered by browser back/forward (URL is already right)\n'
        'if not st.session_state.pop("_url_nav_from_browser", False):\n'
        '    _url_symbol = ""\n'
        '    if page == "\U0001f50d Stock Analysis" and st.session_state.get("_analysis_active"):\n'
        '        _url_symbol = st.session_state.get("_sym_main_result", "")\n'
        '    _write_url_params(page, _url_symbol)',
        '_url_symbol = ""\n'
        'if page == "\U0001f50d Stock Analysis" and st.session_state.get("_analysis_active"):\n'
        '    _url_symbol = st.session_state.get("_sym_main_result", "")\n'
        '_write_url_params(page, _url_symbol)'
    )
    changes += 1
    print("2. Simplified bottom URL sync block")
else:
    print("2. SKIP (bottom block already simplified)")

with open(r"c:\3funspace\stock-analyzer\app.py", "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nDone! {changes} edits applied.")
