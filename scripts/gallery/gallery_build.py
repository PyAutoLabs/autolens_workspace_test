"""
Gallery: Visualization Contact Sheet + Manifest
===============================================

Builds a static HTML contact sheet and a machine-readable manifest of every
figure the visualization scripts have written to disk, so a human (or agent)
can review the full visual output surface in one place.

Walks `scripts/<domain>/images/**` for `.png` (figures) and `.fits` (data
products) produced by the `visualization*.py` scripts and emits:

    output/gallery/gallery.html       — contact sheet, grouped domain → script → group
    output/gallery/viz_manifest.yaml  — {domain: {script: [{file, kind, group}]}}

Both land under `output/` (gitignored). Images are referenced by relative
path, so the gallery is viewable directly from the workspace tree.

Usage (from the workspace root):

    python scripts/gallery/gallery_build.py           # build
    python scripts/gallery/gallery_build.py --check   # build + verify every
                                                      # <img> resolves and the
                                                      # manifest matches disk
    python scripts/gallery/gallery_build.py --embed   # also write a single
                                                      # self-contained
                                                      # gallery_embedded.html
                                                      # (base64 images) that is
                                                      # viewable from anywhere
"""

import argparse
import base64
import html
import sys
from pathlib import Path

import yaml

WORKSPACE_PATH = Path(__file__).resolve().parents[2]
GALLERY_PATH = WORKSPACE_PATH / "output" / "gallery"

CSS = """
body { font-family: sans-serif; margin: 1.5rem; background: #111; color: #ddd; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.2rem; border-bottom: 1px solid #444; padding-bottom: 0.2rem; margin-top: 2rem; }
h3 { font-size: 1.0rem; color: #aaa; margin-top: 1.2rem; }
.grid { display: flex; flex-wrap: wrap; gap: 0.8rem; }
figure { margin: 0; width: 320px; }
figure img { width: 100%; background: #fff; border-radius: 4px; }
figcaption { font-size: 0.75rem; color: #999; word-break: break-all; padding-top: 0.2rem; }
.counts { color: #888; font-size: 0.85rem; }
"""


def scan_images():
    """Manifest dict from the on-disk image trees: domain -> script -> entries.

    `group` is the sub-directory within the script's image dir (e.g. the
    per-source subfolders `parametric` / `rectangular` / `delaunay`), "" at
    the top level.
    """
    manifest = {}
    for images_dir in sorted(WORKSPACE_PATH.glob("scripts/*/images")):
        domain = images_dir.parent.name
        for script_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
            entries = []
            for f in sorted(script_dir.rglob("*")):
                if f.suffix not in (".png", ".fits"):
                    continue
                group = str(f.parent.relative_to(script_dir))
                entries.append(
                    {
                        "file": str(f.relative_to(WORKSPACE_PATH)),
                        "kind": f.suffix.lstrip("."),
                        "group": "" if group == "." else group,
                    }
                )
            if entries:
                manifest.setdefault(domain, {})[script_dir.name] = entries
    return manifest


def render_html(manifest, embed=False):
    n_png = sum(
        1
        for scripts in manifest.values()
        for e in scripts.values()
        for x in e
        if x["kind"] == "png"
    )
    n_fits = sum(
        1
        for scripts in manifest.values()
        for e in scripts.values()
        for x in e
        if x["kind"] == "fits"
    )
    lines = [
        "<meta charset='utf-8'><title>PyAuto Visualization Gallery</title>",
        f"<style>{CSS}</style>",
        "<h1>PyAuto Visualization Gallery</h1>",
        f"<p class='counts'>{n_png} figures (.png) · {n_fits} data products (.fits, listed in the manifest only) · "
        f"source: <code>scripts/&lt;domain&gt;/images/</code></p>",
    ]
    for domain, scripts in manifest.items():
        lines.append(f"<h2>{html.escape(domain)}</h2>")
        for script, entries in scripts.items():
            pngs = [e for e in entries if e["kind"] == "png"]
            if not pngs:
                continue
            lines.append(f"<h3>{html.escape(script)} ({len(pngs)} figures)</h3>")
            lines.append("<div class='grid'>")
            for e in pngs:
                if embed:
                    data = base64.b64encode(
                        (WORKSPACE_PATH / e["file"]).read_bytes()
                    ).decode()
                    src = f"data:image/png;base64,{data}"
                else:
                    src = "../../" + e["file"]
                caption = (e["group"] + "/" if e["group"] else "") + Path(
                    e["file"]
                ).name
                lines.append(
                    f"<figure><img loading='lazy' src='{html.escape(src)}'>"
                    f"<figcaption>{html.escape(caption)}</figcaption></figure>"
                )
            lines.append("</div>")
    return "\n".join(lines) + "\n"


def check(manifest):
    """Verify every gallery <img> target and manifest entry exists on disk (and
    every on-disk image is in the manifest). Returns a list of problems."""
    problems = []
    listed = set()
    for scripts in manifest.values():
        for entries in scripts.values():
            for e in entries:
                listed.add(e["file"])
                if not (WORKSPACE_PATH / e["file"]).is_file():
                    problems.append(f"manifest entry missing on disk: {e['file']}")
    on_disk = {
        str(f.relative_to(WORKSPACE_PATH))
        for f in WORKSPACE_PATH.glob("scripts/*/images/**/*")
        if f.suffix in (".png", ".fits")
    }
    for f in sorted(on_disk - listed):
        problems.append(f"on disk but not in manifest: {f}")
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()

    manifest = scan_images()
    if not manifest:
        print(
            "No images found under scripts/*/images — run scripts/gallery/gallery_run.sh first."
        )
        sys.exit(1)

    GALLERY_PATH.mkdir(parents=True, exist_ok=True)
    (GALLERY_PATH / "viz_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=True)
    )
    (GALLERY_PATH / "gallery.html").write_text(render_html(manifest))
    if args.embed:
        (GALLERY_PATH / "gallery_embedded.html").write_text(
            render_html(manifest, embed=True)
        )

    for domain, scripts in manifest.items():
        for script, entries in scripts.items():
            n_png = sum(1 for e in entries if e["kind"] == "png")
            n_fits = sum(1 for e in entries if e["kind"] == "fits")
            print(f"{domain}/{script}: {n_png} png, {n_fits} fits")
    print(f"\nGallery:  {GALLERY_PATH / 'gallery.html'}")
    print(f"Manifest: {GALLERY_PATH / 'viz_manifest.yaml'}")
    if args.embed:
        print(f"Embedded: {GALLERY_PATH / 'gallery_embedded.html'}")

    if args.check:
        problems = check(manifest)
        if problems:
            print("\nCHECK FAILED:")
            for p in problems:
                print(f"  {p}")
            sys.exit(1)
        print("check: all gallery/manifest entries resolve on disk.")
