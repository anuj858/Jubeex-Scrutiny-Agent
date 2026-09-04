"""
Configuration for the extraction review application.

Adapted for JubeeX Core Filing Record schema with deep descriptive context.
"""

import logging
from typing import Annotated, Any

from llama_cloud.types.beta.split_category import SplitCategory
from llama_cloud.types.classify_v2_parameters import ClassifyV2Parameters, Rule
from llama_cloud.types.extract_v2_parameters import ExtractV2Parameters
from llama_cloud.types.parse_v2_parameters import ParseV2Parameters
from llama_cloud.types.split_v1_parameters import SplitV1Parameters
from pydantic import BaseModel, ConfigDict, Field, BeforeValidator, field_validator, model_validator

from .json_util import create_union_schema as create_union_schema
from .json_util import get_extraction_schema as get_extraction_schema

logger = logging.getLogger(__name__)

EXTRACTED_DATA_COLLECTION: str = "jubeex-filing-extraction"

JUBEEX_FILING_TYPES = [
    "SLP_CIVIL", 
    "SLP_CRIMINAL", 
    "ARBITRATION_PETITION", 
    "WRIT_PETITION_CIVIL", 
    "WRIT_PETITION_CRIMINAL", 
    "other"
]

CONFIDENCE_DESCRIPTION = (
    "Extractor confidence as a percentage string such as 95% or 65%. "
    "Do not use a 0-1 decimal."
)


def coerce_confidence_percent(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("%"):
            try:
                return f"{int(round(float(text[:-1].strip())))}%"
            except ValueError:
                return text
        try:
            number = float(text)
        except ValueError:
            return value
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        return value
    if 0.0 <= number <= 1.0:
        return f"{int(round(number * 100))}%"
    if 0.0 <= number <= 100.0:
        return f"{int(round(number))}%"
    return value


ConfidencePercent = Annotated[str | None, BeforeValidator(coerce_confidence_percent)]


def _unwrap_named(value: Any) -> Any:
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str):
            text = name.strip()
            return text or None
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


class CauseTitle(BaseModel):
    """The 'X versus Y' heading of the case. Extracted as printed."""
    title: str | None = Field(default=None, description="Short cause title as commonly cited, e.g. 'Meera Krishnan v. Union of India & Anr.'")
    formatted_title: str | None = Field(
        default=None,
        description=(
            "Cause title built from Cover Page main names and party counts. "
            "Each side is 'MainName' if that side has 1 party, "
            "'MainName and Anr.' if that side has exactly 2 parties, "
            "'MainName and Ors.' if that side has 3 or more. "
            "Join sides with ' VS '. Example: two petitioners and four respondents → "
            "'Meera Krishnan and Anr. VS Union of India and Ors.' "
            "Main names come from the Cover Page. Do not invent extra parties."
        ),
    )
    raw_text: str | None = Field(default=None, description="The cause title verbatim, including '...and others' / '...and another'")
    main_petitioner: str | None = Field(
        default=None,
        description=(
            "Main petitioner name from the Cover Page cause-title line. "
            "The person's or body's name only. Do not include And Anr, And Ors, "
            "Petitioner, Petitioner(s), or leading dots."
        ),
    )
    main_respondent: str | None = Field(
        default=None,
        description=(
            "Main respondent name from the Cover Page cause-title line. "
            "The person's or body's name only. Do not include And Anr, And Ors, "
            "Respondent, Respondent(s), or leading dots."
        ),
    )
    source_part: str | None = Field(default=None, description="Split label where the stored title was filled (Memo of Parties or Cover Page).")
    source_pages: list[int] = Field(default_factory=list, description="Global page numbers containing this data")
    confidence: ConfidencePercent = Field(default=None, description=CONFIDENCE_DESCRIPTION)


class Party(BaseModel):
    """One petitioner or respondent. Fill from Memo of Parties then Main Petition page 1."""
    serial: int | None = Field(default=None, description="Position in the cause title, 1-based")
    kind: str | None = Field(
        default=None,
        description=(
            "INDIVIDUAL or ORGANIZATION from the printed name on the Main Petition "
            "(and Memo of Parties). ORGANIZATION if the name has a prefix "
            "M/s, M/s., Messrs, The, Union, Government of, Ministry of, Department of, "
            "or a suffix Pvt Ltd, Pvt. Ltd., Private Limited, Ltd, Limited, LLP, LLC, "
            "Inc., Corp., Corporation, Co., Company, Foundation, Trust, Society, Association. "
            "Treat 'The' as a name prefix (The State of …), not every word 'the'. "
            "Always uppercase. Leave null if unclear."
        ),
    )
    name: str | None = Field(default=None, description="Full name as printed. Do not concatenate S/o or W/o into the name.")
    entity_type: str | None = Field(
        default=None,
        description=(
            "Short type label from the printed name, e.g. INDIVIDUAL, Union, Company. "
            "Not limited to a fixed enum. Leave null if not printed."
        ),
    )
    acting_through: str | None = Field(
        default=None,
        description=(
            "Person named after 'acting through' / 'through' under the party block. "
            "May appear under an organization or an individual. "
            "ORGANIZATION almost always has this; if missing, leave null and add an inconsistencies item. "
            "INDIVIDUAL: fill only when those words appear."
        ),
    )
    relation: str | None = Field(default=None, description="Relation prefix only, e.g. S/o, D/o, W/o. Leave null if not printed.")
    guardian: str | None = Field(default=None, description="Related person's name after S/o, D/o, W/o. Leave null if not printed.")
    occupation: str | None = Field(default=None, description="Occupation ONLY if explicitly stated.")
    address: str | None = Field(default=None, description="Verbatim address. Leave null if not printed on a fill source. Never copy from Vakalatnama.")
    city: str | None = Field(default=None, description="Extracted city")
    state: str | None = Field(default=None, description="Extracted state")
    pin_code: str | None = Field(default=None, description="Extracted pincode")
    email: str | None = Field(default=None, description="Email ONLY if explicitly printed. Do not invent.")
    mobile: str | None = Field(default=None, description="Mobile ONLY if explicitly printed. Do not invent.")
    is_primary: bool | None = Field(
        default=None,
        description=(
            "True for the party whose name matches the Cover Page main petitioner or "
            "main respondent. False for the others. Do not invent a party from Cover Page."
        ),
    )
    raw_text: str | None = Field(default=None, description="Verbatim party block as printed, including name, relation, and address lines.")
    source_part: str | None = Field(default=None, description="Must be Memo of Parties or Main Petition. Never Vakalatnama or Cover Page.")
    source_pages: list[int] = Field(default_factory=list, description="Global page numbers containing this data")
    confidence: ConfidencePercent = Field(default=None, description=CONFIDENCE_DESCRIPTION)

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip().upper()
        if text in {"ORGANISATION", "ORG"}:
            return "ORGANIZATION"
        if text in {"INDIVIDUAL", "ORGANIZATION"}:
            return text
        return value


class AdvocateOnRecord(BaseModel):
    """AOR identity. Fill from Vakalatnama then AOR's Declaration."""
    name: str | None = Field(default=None, description="AOR name")
    registration_number: str | None = Field(default=None, description="AOR code / registration number. Extract verbatim.")
    email: str | None = Field(default=None, description="AOR email ONLY if printed")
    mobile: str | None = Field(default=None, description="AOR mobile ONLY if printed")
    firm: str | None = Field(default=None, description="Firm name, if any")
    office_address: str | None = Field(default=None, description="Office address as printed")
    source_part: str | None = Field(default=None, description="Vakalatnama or AOR's Declaration")
    raw_text: str | None = Field(default=None, description="Verbatim AOR block as printed, including name, address, email, and contact.")
    source_pages: list[int] = Field(default_factory=list, description="Global page numbers containing this data")
    confidence: ConfidencePercent = Field(default=None, description=CONFIDENCE_DESCRIPTION)

    @model_validator(mode="before")
    @classmethod
    def coerce_office(cls, value: Any) -> Any:
        if isinstance(value, dict) and not value.get("office_address") and value.get("office"):
            value = dict(value)
            value["office_address"] = value.get("office")
        return value


class ImpugnedOrder(BaseModel):
    """The order under challenge. Primary Impugned Order slot only."""
    model_config = ConfigDict(populate_by_name=True)

    is_primary: bool | None = Field(default=True, description="True for the Impugned Order slot. Do not invent extra annexure orders.")
    case_number: str | None = Field(default=None, description="Case number before the earlier court")
    order_date: str | None = Field(default=None, description="Date of the order under challenge")
    Forum: str | None = Field(default=None, description="The court/forum that passed the impugned order")
    bench: str | None = Field(default=None, description="Bench or seat as printed")
    certified_copy_applied_on: str | None = Field(default=None, description="Date the certified copy was applied for")
    certified_copy_obtained_on: str | None = Field(default=None, description="Date it was obtained")
    raw_text: str | None = Field(default=None, description="Verbatim impugned-order identification as printed.")
    source_part: str | None = Field(default=None, description="Impugned Order")
    source_pages: list[int] = Field(default_factory=list, description="Global page numbers containing this data")
    confidence: ConfidencePercent = Field(default=None, description=CONFIDENCE_DESCRIPTION)

    @model_validator(mode="before")
    @classmethod
    def coerce_forum(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if value.get("Forum") or value.get("forum"):
            return value
        court_name = value.get("court_name")
        if court_name:
            value = dict(value)
            value["Forum"] = court_name
        return value


class InconsistencyItem(BaseModel):
    id: str | None = Field(default=None, description="Sequential id as a string, starting at 1.")
    label: str | None = Field(default=None, description="Short mismatch label")
    raw_text: str | None = Field(default=None, description="What differed, quoting both sources.")

    @model_validator(mode="before")
    @classmethod
    def coerce_detail(cls, value: Any) -> Any:
        if isinstance(value, dict) and not value.get("raw_text") and value.get("detail"):
            value = dict(value)
            value["raw_text"] = value.get("detail")
        return value


class Inconsistencies(BaseModel):
    items: list[InconsistencyItem] = Field(default_factory=list, description="Spelling or value mismatches between fill and verify sources. Also record an ORGANIZATION party with no acting_through.")
    source_part: list[str] = Field(default_factory=list, description="Split labels involved in the mismatches.")
    source_pages: list[int] = Field(default_factory=list, description="Global page numbers involved in the mismatches.")
    confidence: ConfidencePercent = Field(default=None, description=CONFIDENCE_DESCRIPTION)

    @field_validator("source_part", mode="before")
    @classmethod
    def coerce_source_part_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        return value


class DocumentList(BaseModel):
    count: int | None = Field(default=0, description="Count of items")
    items: list[str] = Field(default_factory=list, description="List of the items")


class FilingSummary(BaseModel):
    """High-level summary. Documents are filled by the pipeline from the stitch."""
    matter_title: str | None = Field(default=None, description="Short human title, normally the cause title")
    matter_type: str | None = Field(default=None, description="Petition type label")
    documents: DocumentList | None = Field(default=None, description="List of parts present in the filing")
    annexures: DocumentList | None = Field(default=None, description="Annexures named in the Index")
    estimated_review_minutes: int | None = Field(default=None, description="Estimated human review time based on complexity")


class DocumentSpan(BaseModel):
    name: str | None = Field(default=None, description="Split document part name")
    start_page: int | None = Field(default=None, description="First global page")
    end_page: int | None = Field(default=None, description="Last global page")


class DocumentCounts(BaseModel):
    processed: int | None = Field(default=0)
    failed: int | None = Field(default=0)


class LegalExtractRecord(BaseModel):
    """Fields LlamaExtract fills from the extract pack."""
    court: str | None = Field(
        default=None,
        description=(
            "Court where the petition is filed, as a string. "
            "Fill from Cover Page; check spelling on Main Petition, Vakalatnama, "
            "Office Report on Limitation, Affidavit, and Memo of Parties."
        ),
    )
    petition_type: str | None = Field(
        default=None,
        description=(
            "Petition type as printed, e.g. Special Leave Petition (Civil). "
            "Fill from Cover Page; check spelling on Main Petition, Vakalatnama, "
            "Office Report on Limitation, Affidavit, and Memo of Parties."
        ),
    )
    cause_title: CauseTitle | None = Field(default=None, description="Cause title representing petitioner vs respondent.")
    petitioners: list[Party] = Field(default_factory=list, description="Petitioners. One record per petitioner.")
    respondents: list[Party] = Field(default_factory=list, description="Respondents. One record per respondent.")
    advocates_on_record: list[AdvocateOnRecord] = Field(default_factory=list, description="Advocates-on-Record.")
    impugned_orders: list[ImpugnedOrder] = Field(default_factory=list, description="Primary impugned order only.")
    relief_sort: str | None = Field(
        default=None,
        description=(
            "Main prayer only. Copy from the Main Petition last 2-3 pages under a "
            "heading Main Prayer or Prayer. Leave null if that heading is not printed."
        ),
    )
    inconsistencies: Inconsistencies | None = Field(
        default=None,
        description=(
            "Spelling mismatches between fill and verify sources, plus an ORGANIZATION "
            "party with no acting_through. items[].id is '1', '2', …; items[].raw_text "
            "quotes both sources."
        ),
    )

    @field_validator("court", "petition_type", mode="before")
    @classmethod
    def coerce_named_string(cls, value: Any) -> Any:
        return _unwrap_named(value)

    @model_validator(mode="before")
    @classmethod
    def coerce_legacy_relief(cls, value: Any) -> Any:
        if not isinstance(value, dict) or value.get("relief_sort"):
            return value
        relief = value.get("relief")
        sought = None
        if isinstance(relief, dict):
            sought = relief.get("sought")
        elif isinstance(relief, str):
            sought = relief
        if isinstance(sought, str) and sought.strip():
            value = dict(value)
            value["relief_sort"] = sought.strip()
        return value


class CoreFilingRecord(LegalExtractRecord):
    """Stored compiled-petition JSON. Envelope keys are filled by the pipeline."""
    schema_version: str | None = Field(default=None, description="Leave null. Filled by the pipeline.")
    job_type: str | None = Field(default=None, description="Leave null. Filled by the pipeline.")
    organization_id: str | None = Field(default=None, description="Leave null.")
    workspace_id: str | None = Field(default=None, description="Leave null.")
    user_id: str | None = Field(default=None, description="Leave null.")
    primary_document_id: str | None = Field(default=None, description="Leave null.")
    documents: list[DocumentSpan] = Field(default_factory=list, description="Leave empty. Filled from the stitch.")
    document_counts: DocumentCounts | None = Field(default=None, description="Leave null. Filled from the stitch.")
    filing_summary: FilingSummary | None = Field(default=None, description="Leave null. Document lists may be filled from the stitch.")
    overall_confidence: ConfidencePercent = Field(default=None, description="Leave null. Filled from LlamaExtract confidence scores as a percentage string such as 91%.")
    generated_at: str | None = Field(default=None, description="Leave null. Filled by the pipeline.")


class ExtractionSchema(CoreFilingRecord):
    """Default extraction schema"""
    pass

FILING_SCHEMAS = {
    "SLP_CIVIL": CoreFilingRecord,
    "SLP_CRIMINAL": CoreFilingRecord,
    "ARBITRATION_PETITION": CoreFilingRecord,
    "WRIT_PETITION_CIVIL": CoreFilingRecord,
    "WRIT_PETITION_CRIMINAL": CoreFilingRecord,
    "other": CoreFilingRecord,
}

class ExtractConfig(ExtractV2Parameters):
    configuration_id: str | None = None

class ClassifyConfig(ClassifyV2Parameters):
    rules: list[Rule] = []
    configuration_id: str | None = None

class ParseConfig(ParseV2Parameters):
    configuration_id: str | None = None

class SplitConfig(SplitV1Parameters):
    categories: list[SplitCategory] = []
    configuration_id: str | None = None

class SplitUploadSlot(BaseModel):
    id: str
    label: str
    parts: list[str]
    required: bool = True

class SplitUploadType(BaseModel):
    label: str
    slots: list[SplitUploadSlot] = Field(default_factory=list)
    extract_field_sources: dict[str, Any] | None = None

class SplitUploadConfig(BaseModel):
    extract_field_sources: dict[str, Any] = Field(default_factory=dict)
    types: dict[str, SplitUploadType] = Field(default_factory=dict)

class Config(BaseModel):
    """Root configuration model for configs/config.json."""
    classify: ClassifyConfig
    extract_jubeex: ExtractConfig = Field(alias="extract-jubeex")
    parse: ParseConfig | None = None
    split: SplitConfig | None = None
    split_upload: SplitUploadConfig | None = None
