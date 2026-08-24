"""
Configuration for the extraction review application.

Adapted for JubeeX Core Filing Record schema with deep descriptive context.
"""

import logging
from typing import Optional, List

from llama_cloud.types.beta.split_category import SplitCategory
from llama_cloud.types.classify_v2_parameters import ClassifyV2Parameters, Rule
from llama_cloud.types.extract_v2_parameters import ExtractV2Parameters
from llama_cloud.types.parse_v2_parameters import ParseV2Parameters
from llama_cloud.types.split_v1_parameters import SplitV1Parameters
from pydantic import BaseModel, Field

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

class CauseTitle(BaseModel):
    """The 'X versus Y' heading of the case. Extracted as printed."""
    raw_text: str | None = Field(default=None, description="The cause title verbatim, including '...and others' / '...and another'")
    petitioner_summary: str | None = Field(default=None, description="The petitioner side as printed, e.g. 'M/s Acme Industries Pvt. Ltd. and 2 others'")
    respondent_summary: str | None = Field(default=None, description="The respondent side as printed")
    formatted_title: str | None = Field(default=None, description="A strictly formatted version of the cause title: '1st Petitioner Name [And ors.] VS 1st Respondent Name [And ors.]'. You MUST append 'And ors.' if there are multiple petitioners or multiple respondents.")
    source_part: str | None = Field(default=None, description="The document part where this was found (e.g. COVER_PAGE, PETITION)")
    source_pages: list[str] = Field(default_factory=list, description="Pages containing this data")

class Address(BaseModel):
    """Standardized address format, preserving original text."""
    type: str | None = Field(default=None, description="REGISTERED_OFFICE, RESIDENTIAL, OFFICE, SERVICE, or null")
    full_text: str | None = Field(default=None, description="Mandatory verbatim address text. Do not omit any details.")
    city: str | None = Field(default=None, description="Extracted city")
    state: str | None = Field(default=None, description="Extracted state")
    pin_code: str | None = Field(default=None, description="Extracted pincode")
    country: str | None = Field(default=None, description="Extracted country")

class Petitioner(BaseModel):
    """One record per petitioner."""
    serial: int | None = Field(default=None, description="Position in the cause title, 1-based (e.g. 1)")
    full_name: str | None = Field(default=None, description="Full name as printed")
    entity_type: str | None = Field(default=None, description="INDIVIDUAL, COMPANY, PARTNERSHIP, SOCIETY, TRUST, STATUTORY_BODY, STATE, UNION_OF_INDIA, OTHER")
    acting_through: str | None = Field(default=None, description="The authorised signatory / representative, where the petitioner is not an individual")
    relation_details: str | None = Field(default=None, description="Combined relation and name (e.g. 'W/O Shri Vivek Shah', 'S/O John Doe'). This may be indicated by prefixes like 'S/o', 'D/o', 'W/o', or it may simply be printed directly under the main person's name. Extract the full relationship context.")
    address_full_text: str | None = Field(default=None, description="Mandatory verbatim address text. Do not omit any details.")
    city: str | None = Field(default=None, description="Extracted city")
    state: str | None = Field(default=None, description="Extracted state")
    pin_code: str | None = Field(default=None, description="Extracted pincode")
    source_part: str | None = Field(default=None, description="The document part where this was found")
    source_pages: str | None = Field(default=None, description="Pages containing this data (comma-separated string, e.g. '6, 50, 52')")

class Respondent(BaseModel):
    """One record per respondent."""
    serial: int | None = Field(default=None, description="Position in the cause title, 1-based (e.g. 1)")
    full_name: str | None = Field(default=None, description="Full name as printed. DO NOT concatenate relations like 'S/o'.")
    relation_details: str | None = Field(default=None, description="Combined relation and name (e.g. 'W/O Shri Vivek Shah', 'S/O John Doe'). This may be indicated by prefixes like 'S/o', 'D/o', 'W/o', or it may simply be printed directly under the main person's name. Extract the full relationship context.")
    age: int | None = Field(default=None, description="Age as an integer, ONLY if explicitly stated.")
    occupation: str | None = Field(default=None, description="Occupation ONLY if explicitly stated.")
    address_full_text: str | None = Field(default=None, description="Mandatory verbatim address text. Do not omit any details.")
    city: str | None = Field(default=None, description="Extracted city")
    state: str | None = Field(default=None, description="Extracted state")
    pin_code: str | None = Field(default=None, description="Extracted pincode")
    source_part: str | None = Field(default=None, description="The document part where this was found")
    source_pages: str | None = Field(default=None, description="Pages containing this data (comma-separated string, e.g. '6, 50, 52')")

class MatterClassification(BaseModel):
    """Classification parameters derived typically from the Listing Proforma."""
    main_category: str | None = Field(default=None, description="Registry subject category, e.g. 'Service Matters'")
    sub_category: str | None = Field(default=None, description="Registry sub-category under the main category")
    special_category: str | None = Field(default=None, description="Special category derived from the filing")
    is_pil: bool | None = Field(default=None, description="True if the matter is a Public Interest Litigation")
    source_part: str | None = Field(default=None, description="The document part where this was found")
    source_pages: list[str] = Field(default_factory=list, description="Pages containing this data")

class ImpugnedOrder(BaseModel):
    """The order under challenge and the earlier court."""
    applicable: bool | None = Field(default=True, description="False if this petition type does not have an impugned order (e.g. ARBITRATION_PETITION, WRIT_PETITION_CIVIL).")
    not_applicable_reason: str | None = Field(default=None, description="Reason if applicable is false.")
    date_of_impugned_order: str | None = Field(default=None, description="Date of the order under challenge")
    earlier_court: str | None = Field(default=None, description="The court that passed the impugned order")
    case_number: str | None = Field(default=None, description="Case number before the earlier court")
    petition_type: str | None = Field(default=None, description="The proceeding type before the earlier court, NOT the present one")
    certified_copy_applied_on: str | None = Field(default=None, description="Date the certified copy was applied for")
    certified_copy_obtained_on: str | None = Field(default=None, description="Date it was obtained")
    source_part: str | None = Field(default=None, description="The document part where this was found (usually IMPUGNED_ORDER)")
    source_pages: list[str] = Field(default_factory=list, description="Pages containing this data")

class AdvocateOnRecord(BaseModel):
    """AOR identity and contact details from the AOR Declaration."""
    name: str | None = Field(default=None, description="AOR name")
    registration_number: str | None = Field(default=None, description="AOR code / registration number. Extract verbatim.")
    email: str | None = Field(default=None, description="AOR Email Address")
    mobile: str | None = Field(default=None, description="AOR Mobile Number")
    firm_name: str | None = Field(default=None, description="Firm name, if any")
    office_address: Address | None = Field(default=None, description="Office Address of the AOR")
    source_part: str | None = Field(default=None, description="The document part where this was found (usually AOR_DECLARATION)")
    source_pages: list[str] = Field(default_factory=list, description="Pages containing this data")

class DocumentList(BaseModel):
    count: int | None = Field(default=0, description="Count of items")
    items: list[str] = Field(default_factory=list, description="List of the items")

class FilingSummary(BaseModel):
    """High-level summary of the entire filing bundle."""
    court: str | None = Field(default=None, description="Mirrors the resolved court")
    matter_type: str | None = Field(default=None, description="Mirrors the resolved petition type")
    matter_title: str | None = Field(default=None, description="Short human title for the matter, normally the cause title")
    documents: DocumentList | None = Field(default=None, description="List of parts present in the filing")
    annexures: DocumentList | None = Field(default=None, description="List of annexures present")
    applications: DocumentList | None = Field(default=None, description="List of applications filed alongside")
    estimated_review_minutes: int | None = Field(default=None, description="Estimated human review time based on complexity")
    estimated_review_basis: str | None = Field(default=None, description="How the estimated review time was arrived at")

class CoreFilingRecord(BaseModel):
    """
    Core Filing Record for JubeeX parsing pipeline.
    Same shape for all petition types. Do not omit blocks even if unfillable.
    """
    petition_id: str | None = Field(default=None, description="Unique identifier for the petition.")
    court: str | None = Field(default=None, description="The court where the petition is filed.")
    petition_type: str | None = Field(default=None, description="Type of the petition (e.g. SLP_CIVIL, WRIT_PETITION_CRIMINAL).")
    special_category: str | None = Field(default=None, description="Special category of the petition.")
    
    cause_title: CauseTitle | None = Field(default=None, description="Cause title representing petitioner vs respondent.")
    petitioners: list[Petitioner] = Field(default_factory=list, description="Array of petitioners. One record per petitioner.")
    respondents: list[Respondent] = Field(default_factory=list, description="Array of respondents. One record per respondent.")
    matter_classification: MatterClassification | None = Field(default=None, description="Main category, sub category, special category, PIL flag.")
    impugned_order: ImpugnedOrder | None = Field(default=None, description="The challenged order and the earlier court.")
    advocate_on_record: AdvocateOnRecord | None = Field(default=None, description="AOR identity and contact details.")
    filing_summary: FilingSummary | None = Field(default=None, description="Court, matter type and title, counts, review estimate.")

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

class Config(BaseModel):
    """Root configuration model for configs/config.json."""
    classify: ClassifyConfig
    extract_jubeex: ExtractConfig = Field(alias="extract-jubeex")
    parse: ParseConfig | None = None
    split: SplitConfig | None = None
