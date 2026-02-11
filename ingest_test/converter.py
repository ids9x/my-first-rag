"""Wrap Docling's Classic pipeline with nuclear-optimized settings."""
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from . import config


def build_converter() -> DocumentConverter:
    """Create a configured DocumentConverter for nuclear regulatory PDFs."""
    pipeline_options = PdfPipelineOptions()

    # OCR — needed for scanned IAEA documents
    pipeline_options.do_ocr = config.DO_OCR
    pipeline_options.ocr_options.lang = config.OCR_LANG

    # Table structure — critical for compliance matrices, requirement tables
    pipeline_options.do_table_structure = config.DO_TABLE_STRUCTURE
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=config.TABLE_CELL_MATCHING,
        mode=TableFormerMode.ACCURATE
            if config.TABLE_MODE == "ACCURATE"
            else TableFormerMode.FAST,
    )

    # Hardware acceleration
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=config.NUM_THREADS,
        device=AcceleratorDevice.AUTO,  # will pick CUDA on your Spark
    )

    # Offline model artifacts (air-gapped operation)
    if config.ARTIFACTS_PATH:
        pipeline_options.artifacts_path = str(config.ARTIFACTS_PATH)

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )


def convert_pdf(pdf_path: Path) -> "DoclingDocument":
    """Convert a single PDF and return the DoclingDocument."""
    converter = build_converter()
    result = converter.convert(str(pdf_path))
    return result.document