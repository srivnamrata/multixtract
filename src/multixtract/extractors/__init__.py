"""Pluggable document extractors.

Built-in extractors register themselves on the process-wide ``default_registry``
when this package is imported. To support a new format, implement the
:class:`~multixtract.interfaces.DocumentExtractor` protocol and call
:func:`register_extractor`.
"""
from .docx import DocxExtractor
from .eml import EmlExtractor
from .epub import EpubExtractor
from .excel import ExcelExtractor
from .html import HtmlExtractor
from .image import ImageExtractor
from .legacy import (
    ConvertingExtractor,
    convert_with_libreoffice,
    find_libreoffice,
    legacy_doc_extractor,
    legacy_odp_extractor,
    legacy_ods_extractor,
    legacy_odt_extractor,
    legacy_ppt_extractor,
    legacy_xls_extractor,
)
from .markdown import MarkdownExtractor
from .pdf import PdfExtractor
from .pptx import PptxExtractor
from .registry import (
    ExtractorRegistry,
    default_registry,
    get_extractor,
    register_extractor,
)
from .rtf import RtfExtractor
from .text import TextExtractor

# ---- Register built-in extractors -----------------------------------------
register_extractor(PdfExtractor())
register_extractor(DocxExtractor())  # native .docx
register_extractor(PptxExtractor())  # native .pptx
register_extractor(ExcelExtractor())  # .xlsx / .xlsm / .csv
register_extractor(ImageExtractor())  # .png / .jpg / .jpeg / .tiff / .tif / .webp / .bmp
register_extractor(TextExtractor())      # .txt / .log / .conf / .ini
register_extractor(MarkdownExtractor())  # .md (heading-aware splitting)
register_extractor(HtmlExtractor())      # .html / .htm
register_extractor(EmlExtractor())       # .eml
register_extractor(RtfExtractor())       # .rtf
register_extractor(EpubExtractor())      # .epub
# Legacy binaries: convert via LibreOffice, then delegate. They prefer native
# targets (.docx/.pptx) and fall back to PDF — so .doc now auto-upgrades to
# DocxExtractor and .ppt to PptxExtractor (native, higher fidelity).
register_extractor(legacy_doc_extractor)
register_extractor(legacy_ppt_extractor)
register_extractor(legacy_odt_extractor)  # .odt
register_extractor(legacy_odp_extractor)  # .odp
register_extractor(legacy_ods_extractor)  # .ods
register_extractor(legacy_xls_extractor)  # .xls

__all__ = [
    "ExtractorRegistry",
    "default_registry",
    "get_extractor",
    "register_extractor",
    "PdfExtractor",
    "DocxExtractor",
    "PptxExtractor",
    "ExcelExtractor",
    "ImageExtractor",
    "TextExtractor",
    "MarkdownExtractor",
    "HtmlExtractor",
    "EmlExtractor",
    "RtfExtractor",
    "EpubExtractor",
    "ConvertingExtractor",
    "legacy_doc_extractor",
    "legacy_ppt_extractor",
    "legacy_odt_extractor",
    "legacy_odp_extractor",
    "legacy_ods_extractor",
    "legacy_xls_extractor",
    "convert_with_libreoffice",
    "find_libreoffice",
]
