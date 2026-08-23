"""The UI renders untrusted catalog text; keep it on React's escaped path."""

from pathlib import Path

UI_SRC = Path(__file__).resolve().parents[1] / "ui" / "src"
UI_DIST = Path(__file__).resolve().parents[1] / "ui" / "dist"


def test_ui_source_never_injects_raw_html():
    for path in UI_SRC.glob("*.js*"):
        text = path.read_text()
        assert "dangerouslySetInnerHTML" not in text, path.name
        assert "innerHTML" not in text, path.name
        assert "document.write" not in text, path.name


def test_ui_links_are_filtered_through_safe_url():
    app = (UI_SRC / "App.jsx").read_text()
    assert "safeUrl(item.image_url)" in app and "safeUrl(item.url)" in app
    assert 'rel="noopener noreferrer"' in app


def test_ui_bundle_is_committed():
    assert (UI_DIST / "index.html").exists(), "run `cd ui && npm install && npm run build`"
