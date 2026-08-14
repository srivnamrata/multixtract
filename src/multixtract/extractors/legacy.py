"""Legacy Office format extractors (.doc, .ppt, .xls) via LibreOffice conversion.

The old binary formats (.doc, .ppt) have no reliable native Python parser, so
this module converts them to a modern format with a headless LibreOffice call,
then delegates extraction to the registered extractor for the target format.

Strategy (per the project's design decision):
  * Prefer converting to the modern *native* equivalent (.doc -> .docx,
    .ppt -> .pptx) for the best fidelity and a consistent output schema.
  * Fall back to PDF when no native extractor is registered (the bundled
    PdfExtractor always works), so legacy files are handled out of the box.
  * To force PDF, construct ``ConvertingExtractor(..., targets=(".pdf",))``.

LibreOffice is a SYSTEM dependency (NOT pip-installable). If it is missing,
:meth:`ConvertingExtractor.extract` raises a clear, actionable error telling the
user to install LibreOffice or pre-convert the file to .docx/.pptx themselves.

    apt-get install libreoffice
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .registry import ExtractorRegistry, default_registry


def find_libreoffice() -> Optional[str]:
    """Return the path to the LibreOffice/soffice binary, or None if absent."""
    for name in ("libreoffice", "soffice"):
        path = shutil.which(name)
        if path:
            return path
    return None


def convert_with_libreoffice(
    src_path: str,
    target_ext: str,
    out_dir: str,
    timeout: int = 120,
) -> str:
    """Convert *src_path* to *target_ext* (e.g. ``".docx"`` / ``".pdf"``) inside
    *out_dir* using headless LibreOffice. Returns the converted file path.
    """
    soffice = find_libreoffice()
    if soffice is None:
        raise RuntimeError(
            "LibreOffice is required to process legacy .doc/.ppt files but was "
            "not found. Install it (apt-get install libreoffice) or convert the "
            "file to a modern format (.docx/.pptx) yourself first."
        )
    fmt = target_ext.lstrip(".")
    proc = subprocess.run(
        [soffice, "--headless", "--convert-to", fmt, "--outdir", out_dir, src_path],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"LibreOffice failed to convert {os.path.basename(src_path)} -> {fmt} "
            f"(exit {proc.returncode}): {(proc.stderr or '').strip()[:300]}"
        )
    # LibreOffice normalizes the output filename (spaces → underscores, special
    # chars dropped/replaced), so we can't predict the exact stem. Scan out_dir
    # for any file with the right extension instead of checking a fixed path.
    matches = sorted(
        f for f in os.listdir(out_dir)
        if f.lower().endswith(f".{fmt}")
    )
    if not matches:
        raise RuntimeError(
            f"LibreOffice exited 0 but produced no .{fmt} file in {out_dir!r} "
            f"(converting {os.path.basename(src_path)})"
        )
    return os.path.join(out_dir, matches[0])


class ConvertingExtractor:
    """DocumentExtractor that converts a legacy file, then delegates.

    It tries each extension in *targets* in order, using the first one that has
    a registered extractor, converts the input to that format via LibreOffice,
    and delegates extraction to that target extractor. This means it transparently
    upgrades to native extraction (.docx/.pptx) once those extractors exist,
    while working today via the PDF fallback.
    """

    def __init__(
        self,
        source_exts: Iterable[str],
        targets: Iterable[str] = (".pdf",),
        registry: Optional[ExtractorRegistry] = None,
        timeout: int = 120,
    ) -> None:
        self.extensions: Tuple[str, ...] = tuple(
            e if e.startswith(".") else f".{e}" for e in source_exts
        )
        self.targets: Tuple[str, ...] = tuple(
            t if t.startswith(".") else f".{t}" for t in targets
        )
        self._registry = registry
        self.timeout = timeout

    def _resolve_target(self, reg: ExtractorRegistry):
        """Return ``(target_ext, target_extractor)`` for the first target that
        has a registered extractor."""
        for ext in self.targets:
            try:
                return ext, reg.get(f"_probe{ext}")
            except ValueError:
                continue
        raise RuntimeError(
            f"No extractor registered for any conversion target {self.targets} "
            f"of {self.extensions}. Register a target extractor (e.g. PdfExtractor) first."
        )

    def extract(
        self,
        path: str,
        image_filter: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        reg = self._registry or default_registry
        target_ext, delegate = self._resolve_target(reg)  # fail fast before converting
        with tempfile.TemporaryDirectory(prefix="docextract_legacy_") as tmp:
            converted = convert_with_libreoffice(path, target_ext, tmp, timeout=self.timeout)
            document, prepared = delegate.extract(converted, image_filter=image_filter)
        # Preserve the ORIGINAL file's base name, not the temp converted name.
        document["_base_name"] = os.path.splitext(os.path.basename(path))[0]
        document.setdefault("metadata", {})["converted_from"] = os.path.splitext(path)[1].lower()
        return document, prepared


# Built-in legacy extractors.
#   .doc -> prefer .docx (native), fall back to .pdf
#   .ppt -> prefer .pptx (native), fall back to .pdf
#   .xls -> prefer .xlsx (native), fall back to .pdf
# DocxExtractor, PptxExtractor, and ExcelExtractor are registered in
# extractors/__init__.py, so legacy formats automatically get native high-fidelity
# extraction via LibreOffice conversion. PDF remains the fallback.
legacy_doc_extractor = ConvertingExtractor((".doc",), targets=(".docx", ".pdf"))
legacy_ppt_extractor = ConvertingExtractor((".ppt",), targets=(".pptx", ".pdf"))
legacy_odt_extractor = ConvertingExtractor((".odt",), targets=(".docx", ".pdf"))
legacy_odp_extractor = ConvertingExtractor((".odp",), targets=(".pptx", ".pdf"))
legacy_ods_extractor = ConvertingExtractor((".ods",), targets=(".xlsx", ".pdf"))
legacy_xls_extractor = ConvertingExtractor((".xls",), targets=(".xlsx", ".pdf"))
