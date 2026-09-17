"""Checksum-pinned synthetic reference sheets (advocate / deponent / notary)."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ALLOWED_CLASSES = {"notary_seal", "signature_like_mark"}
ALLOWED_ROLES = {"advocate", "deponent", "petitioner", "respondent", "unknown"}


@dataclass(frozen=True)
class ReferenceAsset:
    reference_id: str
    class_name: str
    expected_role: str
    description: str
    content: bytes


class ReferenceCatalogError(RuntimeError):
    """Raised when an approved reference catalogue fails its safety checks."""


class ReferenceCatalog:
    """Loads only checksum-pinned synthetic/sanitized few-shot reference sheets."""

    def __init__(self, root: Path | None = None):
        self.root = root or Path(__file__).resolve().parent
        self.manifest_path = self.root / "manifest.json"
        self.assets_dir = self.root / "assets"
        self._manifest = self._load_and_validate()
        self.version = str(self._manifest.get("catalog_version") or "")

    def _load_and_validate(self) -> dict[str, Any]:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ReferenceCatalogError(
                "Visual reference manifest is unreadable"
            ) from exc
        if manifest.get("schema_version") != "1.0":
            raise ReferenceCatalogError("Unsupported visual reference manifest schema")
        privacy = manifest.get("privacy_policy") or {}
        if not privacy.get("contains_only_synthetic_or_sanitized_assets"):
            raise ReferenceCatalogError(
                "Reference catalogue is not approved as sanitized"
            )
        if privacy.get("authenticity_examples") or privacy.get("identity_examples"):
            raise ReferenceCatalogError(
                "Identity/authenticity references are prohibited"
            )
        assets = manifest.get("assets")
        if not isinstance(assets, list) or not assets:
            raise ReferenceCatalogError(
                "Reference catalogue contains no approved assets"
            )
        seen: set[str] = set()
        for item in assets:
            reference_id = str(item.get("reference_id") or "")
            if not reference_id or reference_id in seen:
                raise ReferenceCatalogError(
                    "Reference IDs must be non-empty and unique"
                )
            seen.add(reference_id)
            if item.get("class_name") not in ALLOWED_CLASSES:
                raise ReferenceCatalogError(
                    f"Unsupported reference class: {reference_id}"
                )
            if item.get("expected_role") not in ALLOWED_ROLES:
                raise ReferenceCatalogError(
                    f"Unsupported reference role: {reference_id}"
                )
            if not item.get("approved_for_model_use") or item.get(
                "contains_personal_data"
            ):
                raise ReferenceCatalogError(
                    f"Reference is not approved: {reference_id}"
                )
            path = self.assets_dir / str(item.get("filename") or "")
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise ReferenceCatalogError(
                    f"Reference asset is missing: {reference_id}"
                ) from exc
            checksum = hashlib.sha256(content).hexdigest()
            if checksum != item.get("sha256"):
                raise ReferenceCatalogError(
                    f"Reference checksum mismatch: {reference_id}"
                )
            if not content.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ReferenceCatalogError(f"Reference must be a PNG: {reference_id}")
        return manifest

    def all_assets(self) -> list[ReferenceAsset]:
        selected: list[ReferenceAsset] = []
        for item in self._manifest["assets"]:
            path = self.assets_dir / str(item.get("filename") or "")
            cues = ", ".join(str(value) for value in item.get("visual_cues", []))
            nearby = ", ".join(
                str(value) for value in item.get("expected_nearby_wording", [])
            )
            description = (
                f"Synthetic {item['class_name']} contact sheet for role "
                f"{item['expected_role']}. Visual cues: {cues}. "
                f"Nearby wording examples: {nearby}."
            )
            selected.append(
                ReferenceAsset(
                    reference_id=str(item["reference_id"]),
                    class_name=str(item["class_name"]),
                    expected_role=str(item["expected_role"]),
                    description=description,
                    content=path.read_bytes(),
                )
            )
        return selected


def load_reference_assets() -> list[ReferenceAsset]:
    try:
        return ReferenceCatalog().all_assets()
    except ReferenceCatalogError:
        logger.warning("Visual reference sheets unavailable; continuing without them")
        return []
