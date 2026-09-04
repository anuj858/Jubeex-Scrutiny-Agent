"""
Configuration for the extraction review application.

Adapted for JubeeX Core Filing Record schema with deep descriptive context.
"""

import logging
from typing import Any

from llama_cloud.types.beta.split_category import SplitCategory
from llama_cloud.types.classify_v2_parameters import ClassifyV2Parameters, Rule
from llama_cloud.types.extract_v2_parameters import ExtractV2Parameters
from llama_cloud.types.parse_v2_parameters import ParseV2Parameters
from llama_cloud.types.split_v1_parameters import SplitV1Parameters
from pydantic import BaseModel, Field, model_validator

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

class NamedValue(BaseModel):
    """A labelled string such as court or petition type."""
    name: str | None = Field(default=None, description="Printed name. Leave null if not printed. Do not invent.")

    @model_validator(mode="before")
    @classmethod
    def coerce_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return {"name": text or None}
        return value


class CauseTitle(BaseModel):
    """The 'X versus Y' heading of the case. Extracted as printed."""
    title: str | None = Field(default=None, description="Short cause title as commonly cited, e.g. 'Meera Krishnan v. Union of India & Anr.'")
    formatted_title: str | None = Field(default=None, description="Formatted cause title: '1st Petitioner Name [And ors.] VS 1st Respondent Name [And ors.]'. Append 'And ors.' only when the source text shows more than one petitioner or respondent. Do not invent extra parties.")
    raw_text: str | None = Field(default=None, description="The cause title verbatim, including '...and others' / '...and another'")
    source_part: str | None = Field(default=None, description="Split label where the stored title was filled (Memo of Parties or Cover Page).")
    source_pages: list[int] = Field(default_factory=list, description="Global page numbers containing this data")


class Party(BaseModel):
    """One petitioner or respondent. Fill from Memo of Parties then Main Petition page 1."""
    serial: int | None = Field(default=None, description="Position in the cause title, 1-based")
    kind: str | None = Field(default=None, description="individual or organization. Leave null if unclear.")
    name: str | None = Field(default=None, description="Full name as printed. Do not concatenate S/o or W/o into the name.")
    entity_type: str | None = Field(default=None, description="INDIVIDUAL, COMPANY, PARTNERSHIP, SOCIETY, TRUST, STATUTORY_BODY, STATE, UNION_OF_INDIA, OTHER. Leave null if not printed.")
    acting_through: str | None = Field(default=None, description="Authorised signatory / representative when the party is not an individual")
    relation: str | None = Field(default=None, description="Relation prefix only, e.g. S/o, D/o, W/o. Leave null if not printed.")
    guardian: str | None = Field(default=None, description="Related person's name after S/o, D/o, W/o. Leave null if not printed.")
    age: int | None = Field(default=None, description="Age as an integer ONLY if explicitly stated.")
    occupation: str | None = Field(default=None, description="Occupation ONLY if explicitly stated.")
    address: str | None = Field(default=None, description="Verbatim address. Leave null if not printed on a fill source. Never copy from Vakalatnama.")
    city: str | None = Field(default=None, description="Extracted city")
    state: str | None = Field(default=None, description="Extracted state")
    pin_code: str | None = Field(default=None, description="Extracted pincode")
    email: str | None = Field(default=None, description="Email ONLY if explicitly printed. Do not invent.")
    mobile: str | None = Field(default=None, description="Mobile ONLY if explicitly printed. Do not invent.")
    is_primary: bool | None = Field(default=None, description="True for the first named party on that side")
    source_part: str | None = Field(default=None, description="Must be Memo of Parties or Main Petition. Never Vakalatnama or Cover Page.")
    source_pages: list[int] = Field(default_factory=list, description="Global page numbers containing this data")


class AdvocateOnRecord(BaseModel):
    """AOR identity. Fill from Vakalatnama then AOR's Declaration."""
    name: str | None = Field(default=None, description="AOR name")
    registration_number: str | None = Field(default=None, description="AOR code / registration number. Extract verbatim.")
    email: str | None = Field(default=None, description="AOR email ONLY if printed")
    mobile: str | None = Field(default=None, description="AOR mobile ONLY if printed")
    firm: str | None = Field(default=None, description="Firm name, if any")
    office: str | None = Field(default=None, description="Office address as printed")
    source_part: str | None = Field(default=None, description="Vakalatnama or AOR's Declaration")
    source_pages: list[int] = Field(default_factory=list, description="Global page numbers containing this data")


class Classification(BaseModel):
    """Listing Proforma subject classification."""
    main_category_code: str | None = Field(default=None, description="Registry main category code as printed")
    main_category_name: str | None = Field(default=None, description="Registry main category name as printed")
    sub_category_code: str | None = Field(default=None, description="Registry sub-category code as printed")
    sub_category_name: str | None = Field(default=None, description="Registry sub-category name as printed")
    special_categories: list[str] = Field(default_factory=list, description="Special categories as printed on the Listing Proforma. Copy verbatim. Empty if none.")
    is_pil: bool | None = Field(default=None, description="True if the matter is printed as a Public Interest Litigation")
    source_part: str | None = Field(default=None, description="Listing Proforma")
    source_pages: list[int] = Field(default_factory=list, description="Global page numbers containing this data")


class ImpugnedOrder(BaseModel):
    """The order under challenge. Primary Impugned Order slot only."""
    is_primary: bool | None = Field(default=True, description="True for the Impugned Order slot. Do not invent extra annexure orders.")
    case_number: str | None = Field(default=None, description="Case number before the earlier court")
    order_date: str | None = Field(default=None, description="Date of the order under challenge")
    court_name: str | None = Field(default=None, description="The court that passed the impugned order")
    petition_type_name: str | None = Field(default=None, description="The proceeding type before the earlier court, NOT the present one")
    bench: str | None = Field(default=None, description="Bench or seat as printed")
    judges: list[str] = Field(default_factory=list, description="Judge names ONLY if printed. Do not invent.")
    lower_court_name: str | None = Field(default=None, description="Court below the impugned court, if printed")
    certified_copy_applied_on: str | None = Field(default=None, description="Date the certified copy was applied for")
    certified_copy_obtained_on: str | None = Field(default=None, description="Date it was obtained")
    source_part: str | None = Field(default=None, description="Impugned Order")
    source_pages: list[int] = Field(default_factory=list, description="Global page numbers containing this data")


class Relief(BaseModel):
    """Prayer from the end of the Main Petition."""
    sought: str | None = Field(default=None, description="Main prayer as printed")
    interim: str | None = Field(default=None, description="Interim relief as printed")
    source_part: str | None = Field(default=None, description="Main Petition")
    source_pages: list[int] = Field(default_factory=list, description="Global page numbers containing this data")


class ApplicationItem(BaseModel):
    name: str | None = Field(default=None, description="Application name as listed in the Index")
    pages: list[int] = Field(default_factory=list, description="Page numbers from the Index")


class Applications(BaseModel):
    count: int | None = Field(default=0, description="Count of application items")
    items: list[ApplicationItem] = Field(default_factory=list, description="Applications listed in the Index")


class InconsistencyItem(BaseModel):
    id: str | None = Field(default=None, description="Short id such as i1")
    label: str | None = Field(default=None, description="Short mismatch label")
    detail: str | None = Field(default=None, description="What differed, quoting both sources")


class Inconsistencies(BaseModel):
    items: list[InconsistencyItem] = Field(default_factory=list, description="Spelling or value mismatches between fill and verify sources")
    summary: str | None = Field(default=None, description="One-line summary of mismatches. Null if none.")


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
    court: NamedValue | None = Field(default=None, description="Court where the petition is filed.")
    petition_type: NamedValue | None = Field(default=None, description="Petition type as printed, e.g. Special Leave Petition (Civil).")
    cause_title: CauseTitle | None = Field(default=None, description="Cause title representing petitioner vs respondent.")
    petitioners: list[Party] = Field(default_factory=list, description="Petitioners. One record per petitioner.")
    respondents: list[Party] = Field(default_factory=list, description="Respondents. One record per respondent.")
    advocates_on_record: list[AdvocateOnRecord] = Field(default_factory=list, description="Advocates-on-Record.")
    classification: Classification | None = Field(default=None, description="Listing Proforma classification.")
    impugned_orders: list[ImpugnedOrder] = Field(default_factory=list, description="Primary impugned order only.")
    relief: Relief | None = Field(default=None, description="Prayer from the Main Petition.")
    applications: Applications | None = Field(default=None, description="Applications listed in the Index.")
    inconsistencies: Inconsistencies | None = Field(default=None, description="Spelling mismatches between fill and verify sources.")
    filing_summary: FilingSummary | None = Field(default=None, description="Matter title and document lists.")


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
    overall_confidence: float | None = Field(default=None, description="Leave null. Filled from LlamaExtract confidence scores.")
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
