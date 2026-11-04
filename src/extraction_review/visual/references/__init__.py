"""Approved, privacy-safe visual references used only as classification guidance."""

from .catalog import (
    ReferenceAsset,
    ReferenceCatalog,
    ReferenceCatalogError,
    load_reference_assets,
)

__all__ = [
    "ReferenceAsset",
    "ReferenceCatalog",
    "ReferenceCatalogError",
    "load_reference_assets",
]
