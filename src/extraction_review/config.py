"""
Configuration for the extraction review application.

Adapted for JubeeX Core Filing Record schema with deep descriptive context.
"""

import json
import logging
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from llama_cloud.types.beta.split_category import SplitCategory
from llama_cloud.types.classify_v2_parameters import ClassifyV2Parameters, Rule
from llama_cloud.types.extract_v2_parameters import ExtractV2Parameters
from llama_cloud.types.parse_v2_parameters import ParseV2Parameters
from llama_cloud.types.split_v1_parameters import SplitV1Parameters
from pydantic import BaseModel, ConfigDict, Field, BeforeValidator, field_validator, model_validator

from .json_util import create_union_schema as create_union_schema
from .json_util import get_extraction_schema as get_extraction_schema

logger = logging.getLogger(__name__)

_REPO_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_repo_dotenv() -> Path:
    """Load the repo `.env` over process env so local runs match the file."""
    load_dotenv(_REPO_ENV_PATH, override=True)
    return _REPO_ENV_PATH


load_repo_dotenv()

EXTRACTED_DATA_COLLECTION: str = "jubeex-filing-extraction"

JUBEEX_FILING_TYPES = [
    "SLP_CIVIL",
    "SLP_CRIMINAL",
    "CIVIL_APPEAL",
    "CRIMINAL_APPEAL",
    "WRIT_PETITION_CIVIL",
    "WRIT_PETITION_CRIMINAL",
    "TRANSFER_PETITION_CIVIL",
    "TRANSFER_PETITION_CRIMINAL",
    "REVIEW_PETITION_CIVIL",
    "REVIEW_PETITION_CRIMINAL",
    "ORIGINAL_SUIT_CIVIL",
    "CONTEMPT_PETITION_CIVIL",
    "CONTEMPT_PETITION_CRIMINAL",
    "ELECTION_PETITION_CIVIL",
    "ARBITRATION_PETITION",
    "CURATIVE_PETITION_CIVIL",
    "CURATIVE_PETITION_CRIMINAL",
    "MISCELLANEOUS_APPLICATION",
]
JUBEEX_UPLOAD_FILING_TYPES = tuple(JUBEEX_FILING_TYPES)

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
            "Cause title from the fullest party list in the opening pages. "
            "Split labels are not known yet, so the list may be on any of those pages. "
            "Each side is 'MainName' if that side has 1 party, "
            "'MainName and Anr.' if that side has exactly 2 parties, "
            "'MainName and Ors.' if that side has 3 or more. "
            "Join sides with ' VS '. Example: two petitioners and four respondents → "
            "'Meera Krishnan and Anr. VS Union of India and Ors.' "
            "Never write [And ors.], [and Anr.], or any square brackets around "
            "and Anr./and Ors. A caption line that says '& Anr.' or '& Ors.' is a "
            "count hint, not the second party's name. Do not invent extra parties."
        ),
    )
    raw_text: str | None = Field(default=None, description="The cause title verbatim, including '...and others' / '...and another'")
    main_petitioner: str | None = Field(
        default=None,
        description=(
            "First petitioner's name from the fullest numbered party list in the "
            "opening pages. The person's or body's name only. Do not include "
            "And Anr, And Ors, Petitioner, Petitioner(s), or leading dots. "
            "A short caption may print only this name plus And Anr/Ors; the other "
            "names are on another opening page."
        ),
    )
    main_respondent: str | None = Field(
        default=None,
        description=(
            "First respondent's name from the fullest numbered party list in the "
            "opening pages. The person's or body's name only. Do not include "
            "And Anr, And Ors, Respondent, Respondent(s), or leading dots. "
            "A short caption may print only this name plus And Anr/Ors; the other "
            "names and the address are on another opening page."
        ),
    )
    source_part: str | None = Field(default=None, description="Opening pages or Closing pages. Record source_pages for the page that printed this.")
    source_pages: list[int] = Field(default_factory=list, description="Global page numbers containing this data")
    confidence: ConfidencePercent = Field(default=None, description=CONFIDENCE_DESCRIPTION)


class Party(BaseModel):
    """One numbered party from the petition opening. Copy the relation, occupation, and address printed under the name. And Anr or And Ors is not a party. Use only the current-court column when the schedule also has a lower-court column."""
    serial: int | None = Field(default=None, description="Position in the cause title, 1-based")
    kind: str | None = Field(
        default=None,
        description=(
            "INDIVIDUAL or ORGANIZATION from the printed name on any opening page. "
            "ORGANIZATION if the name has a prefix "
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
    address: str | None = Field(default=None, description="Verbatim address printed with this party on any opening page. Leave null if no address is printed. Do not invent one, and do not copy an advocate's chamber.")
    city: str | None = Field(default=None, description="Extracted city")
    state: str | None = Field(default=None, description="Extracted state")
    pin_code: str | None = Field(default=None, description="Extracted pincode")
    email: str | None = Field(default=None, description="Email ONLY if explicitly printed. Do not invent.")
    mobile: str | None = Field(default=None, description="Mobile ONLY if explicitly printed. Do not invent.")
    is_primary: bool | None = Field(
        default=None,
        description=(
            "True for petitioner 1 and respondent 1. False for every later party. "
            "Do not invent a party from a caption that only says And Anr or And Ors."
        ),
    )
    raw_text: str | None = Field(default=None, description="Verbatim party block as printed, including name, relation, and address lines.")
    source_part: str | None = Field(
        default=None,
        description=(
            "Opening pages. Record source_pages for the page that printed the name and address."
        ),
    )
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


FRONT_PETITIONERS_DESCRIPTION = (
    "Every numbered petitioner printed near IN THE MATTER OF, BETWEEN, VERSUS, "
    "POSITION OF THE PARTIES, or MEMO OF PARTIES, including the lines under the name "
    "and the same list continued on the next page. "
    "Look for: 1. 2. 3. with the relation, guardian, age, occupation, and address on the "
    "following lines. Do not stop at a page break and do not skip a printed number. "
    "A party may be only a name and a place; no S/o and no Through is still a party. "
    "Presently at, Second Address, and Also at are more place lines of the party above, "
    "not another party. R/o marks the place; do not put R/o or the S/o name into the party name. "
    "A caption that only says '& Anr.' or '& Ors.' is a count, not a party. "
    "When one page has two columns (Before High Court beside Before Supreme Court, or "
    "Trial Court / High Court / In this Court), copy only the current-court column. "
    "Not a Party and Dead are column roles, not people. A case number is not a party. "
    "Do not stop after the first name. Record source_pages."
)
FRONT_RESPONDENTS_DESCRIPTION = (
    "Every numbered respondent printed on the other side of VERSUS, or under RESPONDENT, "
    "in the same block as the petitioners (IN THE MATTER OF, BETWEEN, POSITION OF THE PARTIES, "
    "or MEMO OF PARTIES), including the lines under the name and the same list continued on the next page. "
    "Look for: 1. 2. 3. with the relation, guardian, age, occupation, and address on the following lines. "
    "Do not stop at a page break and do not skip a printed number. "
    "A party may be only a name and a place, such as Land Acquisition Collector. "
    "No S/o and no Through is still a party. "
    "Presently at, Second Address, and Also at are more place lines of the party above, "
    "not another respondent. R/o marks the place; do not put R/o or the S/o name into the party name. "
    "'& Ors.' is not a party. "
    "When one page has two columns (Before High Court beside Before Supreme Court, or "
    "Trial Court / High Court / In this Court), copy only the current-court column. "
    "Not a Party and Dead are column roles, not people. A case number is not a party. "
    "Do not stop after the first name. Record source_pages."
)
FRONT_ADDRESS_DESCRIPTION = (
    "The place lines printed under this party's number, including a continuation on the next page. "
    "Look for: R/o, resident of, residing at, presently at, second address, also at, Gut No., "
    "registered address, having office at, Plot No., city, district, and PIN. "
    "Join those lines into one address. Presently at and Second Address belong to this party. "
    "Keep the place and do not keep a leading R/o. Leave null when that block prints no place. "
    "Do not copy an advocate's chamber. Do not continue into THE HUMBLE APPEAL OF, "
    "MOST RESPECTFULLY SHOWETH, or the prayer."
)
FRONT_OCCUPATION_DESCRIPTION = (
    "The occupation printed under this party's name. "
    "Look for: the word occupation, or a short descriptor with no place line, such as Indian Jew. "
    "Leave null when neither is printed."
)
FRONT_AGE_DESCRIPTION = (
    "The age printed under this party's name. "
    "Look for: 'age 52 years' or 'aged 56 years'. Copy the number and the word years, "
    "for example 52 years. Leave null when no age is printed."
)
FRONT_RELATION_DESCRIPTION = (
    "The relation prefix printed under this party's name. "
    "Look for: S/o, D/o, or W/o. Leave null when not printed."
)
FRONT_GUARDIAN_DESCRIPTION = (
    "The related person's name printed after S/o, D/o, or W/o. "
    "Look for: the name on that same line, such as Gopinath Sharma after S/o. "
    "Leave null when not printed. Do not include the occupation or the address."
)
FRONT_ACTING_THROUGH_DESCRIPTION = (
    "The person this party acts through, printed under the name. "
    "Look for: through, represented by, or rep by, such as through its Chairman "
    "or rep by inspector of police. Leave null when those words are not printed."
)


class FrontParty(Party):
    """One numbered party from the opening pages. Not used by /v1/filings/extract."""

    acting_through: str | None = Field(default=None, description=FRONT_ACTING_THROUGH_DESCRIPTION)
    relation: str | None = Field(default=None, description=FRONT_RELATION_DESCRIPTION)
    guardian: str | None = Field(default=None, description=FRONT_GUARDIAN_DESCRIPTION)
    occupation: str | None = Field(default=None, description=FRONT_OCCUPATION_DESCRIPTION)
    age: str | None = Field(default=None, description=FRONT_AGE_DESCRIPTION)
    address: str | None = Field(default=None, description=FRONT_ADDRESS_DESCRIPTION)


class AdvocateOnRecord(BaseModel):
    """Printed AOR footer only. Leave null if not printed. Do not invent."""
    name: str | None = Field(
        default=None,
        description="AOR name as printed in the AOR footer. Not a party name. Leave null if not printed.",
    )
    registration_number: str | None = Field(
        default=None,
        description="AOR code / CC No. as printed. Leave null if not printed. Do not invent.",
    )
    email: str | None = Field(
        default=None,
        description="AOR email ONLY if printed. Leave null otherwise. Do not invent.",
    )
    mobile: str | None = Field(
        default=None,
        description=(
            "The phone number printed for this advocate. Look for Ph., Phone, Tel., "
            "or Mobile on the chamber line, including a number after the PIN such as "
            "Ph.: 23386175. Copy that number here. Leave null only when no phone is "
            "printed. Do not invent one."
        ),
    )
    firm: str | None = Field(
        default=None,
        description="Firm name if printed in the AOR footer. Leave null otherwise.",
    )
    office_address: str | None = Field(
        default=None,
        description=(
            "Printed Chamber/office lines under the AOR name. Copy the place. Often "
            "labelled Chamber: not Address. Include PIN if printed. Stop before Ph., "
            "Phone, Tel., or Mobile; that number belongs in mobile. Leave null if "
            "those lines are not printed. Never invent a chamber number, block, or PIN. "
            "Never use a party's address. The chamber may be on any opening page or "
            "on the closing pages, under the advocate name after Drawn By / Filed on "
            "/ DRAWN & FILED BY."
        ),
    )
    source_part: str | None = Field(
        default=None,
        description=(
            "Opening pages or Closing pages, whichever printed this footer. Record source_pages."
        ),
    )
    raw_text: str | None = Field(
        default=None,
        description="Verbatim printed AOR footer (name, chamber, email, contact). Do not reconstruct.",
    )
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
    """The order under challenge. Its court, date, and case number may be printed on any opening page."""
    model_config = ConfigDict(populate_by_name=True)

    is_primary: bool | None = Field(default=True, description="True for the Impugned Order slot. Do not invent extra annexure orders.")
    case_number: str | None = Field(default=None, description="Case number before the earlier court")
    order_date: str | None = Field(default=None, description="Date of the order under challenge, printed with its case number on any opening page. Do not use a date that is only quoted inside the prayer. Leave null if the challenged order is not identified.")
    Forum: str | None = Field(default=None, description="The court/forum that passed the impugned order")
    bench: str | None = Field(
        default=None,
        description=(
            "Judge names printed under CORAM on any opening page, preserving "
            "honorifics. Join multiple judges with '; '. Do not put the city, "
            "court seat, place of decision, or forum here."
        ),
    )
    certified_copy_applied_on: str | None = Field(default=None, description="Date the certified copy was applied for")
    certified_copy_obtained_on: str | None = Field(default=None, description="Date it was obtained")
    raw_text: str | None = Field(default=None, description="Verbatim impugned-order identification as printed.")
    source_part: str | None = Field(
        default=None,
        description=(
            "Opening pages. Record source_pages for the page that printed these particulars."
        ),
    )
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
            "It may be printed on any opening page."
        ),
    )
    petition_type: str | None = Field(
        default=None,
        description=(
            "Petition type as printed, e.g. Special Leave Petition (Civil), Civil Appeal, "
            "Writ Petition (Criminal), or Miscellaneous Application. "
            "It may be printed on any opening page."
        ),
    )
    cause_title: CauseTitle | None = Field(
        default=None,
        description=(
            "Cause title representing petitioner vs respondent. "
            "Read it from whichever opening page prints the fullest caption."
        ),
    )
    petitioners: list[Party] = Field(default_factory=list, description="Petitioners. One record per petitioner.")
    respondents: list[Party] = Field(default_factory=list, description="Respondents. One record per respondent.")
    advocates_on_record: list[AdvocateOnRecord] = Field(
        default_factory=list,
        description=(
            "Printed advocate-on-record footer only. Leave a field null if it is not "
            "printed; do not invent name, code, email, mobile, firm, chamber, or PIN. "
            "Do not copy a party's address. The footer may be on any opening page or "
            "on the closing pages."
        ),
    )
    impugned_orders: list[ImpugnedOrder] = Field(
        default_factory=list,
        description=(
            "Primary order under challenge. Its court, date, and case number may be "
            "printed on any opening page. Keep one consistent set."
        ),
    )
    relief_sort: str | None = Field(
        default=None,
        description=(
            "Main prayer body from the opening pages. "
            "Copy the text under the heading Main Prayer or Prayer. "
            "Do not include the heading, clause number, markdown, or HTML "
            "(for example omit '7. MAIN PRAYER:' and '<u>**MAIN PRAYER**</u>:'). "
            "Start at the prayer sentence, e.g. 'In the circumstances stated above…'. "
            "Leave null if that heading is not printed in the opening window."
        ),
    )
    petition_date: str | None = Field(
        default=None,
        description=(
            "Date of this petition from the opening pages, after the heading Main "
            "Prayer or Prayer. Use the date printed next to Date, Dated, Date Drafted, "
            "or Date Filed on. If both Date Drafted and Date Filed on are printed, "
            "use only the Date Drafted date. A page number between the label and the "
            "date is not the date. Return the date as printed, without the label. "
            "Do not use a date inside the prayer or an impugned-order date. "
            "Leave null if none of those labels is printed after the prayer."
        ),
    )
    inconsistencies: Inconsistencies | None = Field(
        default=None,
        description=(
            "Spelling or value mismatches between two printed copies of the same "
            "fact in the opening pages, plus an ORGANIZATION party with no "
            "acting_through. Compare the cause-title main petitioner and main "
            "respondent with petitioner 1 and respondent 1. And Anr or And Ors on "
            "a short caption is not a second party and is not a mismatch. Also "
            "record a mismatch of the challenged order's case number, date, or "
            "forum when those particulars are printed more than once. Also record "
            "a mismatch of the advocate chamber address when it is printed more "
            "than once. Quote both printed strings. Do not invent an address. "
            "Ignore an extra Ph. on the same chamber line. items[].id is '1', '2', …."
        ),
    )

    @field_validator("court", "petition_type", mode="before")
    @classmethod
    def coerce_named_string(cls, value: Any) -> Any:
        return _unwrap_named(value)

    @model_validator(mode="before")
    @classmethod
    def coerce_null_lists_and_relief(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        for key in (
            "impugned_orders",
            "petitioners",
            "respondents",
            "advocates_on_record",
            "documents",
        ):
            if value.get(key) is None:
                value = dict(value)
                value[key] = []
        if value.get("relief_sort"):
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


class FrontExtractRecord(LegalExtractRecord):
    """Extract schema for the opening pages plus the last 5. Not used by /v1/filings/extract."""

    court: str | None = Field(
        default=None,
        description=(
            "Court where the petition is filed (e.g. Supreme Court of India, High Court of Delhi), "
            "as printed on any opening page."
        ),
    )
    petition_type: str | None = Field(
        default=None,
        description=(
            "Petition type (e.g. Special Leave Petition (Civil), Special Leave Petition (Crl.), "
            "Writ Petition (Civil), Transfer Petition), as printed on any opening page."
        ),
    )
    special_category: str | None = Field(
        default=None,
        description=(
            "Matter subtype or subject classification printed in the filing, "
            "such as Tax Matters. Leave null when not explicitly printed."
        ),
    )
    cause_title: CauseTitle | None = Field(
        default=None,
        description=(
            "Cause title from the filing. Formatted title should be '<Petitioner Name(s)> VS <Respondent Name(s)>'. "
            "Also extract main_petitioner and main_respondent."
        ),
    )
    petitioners: list[FrontParty] = Field(
        default_factory=list,
        description=FRONT_PETITIONERS_DESCRIPTION,
    )
    respondents: list[FrontParty] = Field(
        default_factory=list,
        description=FRONT_RESPONDENTS_DESCRIPTION,
    )
    advocates_on_record: list[AdvocateOnRecord] = Field(
        default_factory=list,
        description=(
            "Advocate on record from any opening page or the closing pages. "
            "Copy the printed name, code, chamber, email, and phone. Do not invent them."
        ),
    )
    impugned_orders: list[ImpugnedOrder] = Field(
        default_factory=list,
        description=(
            "Order under challenge. Extract court name, order date, and case number "
            "from whichever opening page prints them."
        ),
    )
    relief_sort: str | None = Field(
        default=None,
        description=(
            "Prayer body under Main Prayer or Prayer on any opening page. "
            "Omit the heading. Leave null only when that prayer is not in the window."
        ),
    )
    petition_date: str | None = Field(
        default=None,
        description=(
            "Date after Main Prayer or Prayer on any opening page, next to Date, "
            "Dated, Date Drafted, or Date Filed on. Date Drafted wins when both "
            "are printed. A page number between the label and the date is not the "
            "date. Leave null only when no such date is printed in the window."
        ),
    )


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
    overall_confidence: ConfidencePercent = Field(default=None, description="Leave null. Filled from LlamaExtract confidence scores as a percentage string such as 91%.")
    generated_at: str | None = Field(default=None, description="Leave null. Filled by the pipeline.")


class ExtractionSchema(CoreFilingRecord):
    """Default extraction schema"""
    pass

FILING_SCHEMAS = {filing_type: CoreFilingRecord for filing_type in JUBEEX_FILING_TYPES}

class ExtractConfig(ExtractV2Parameters):
    configuration_id: str | None = None

class ClassifyConfig(ClassifyV2Parameters):
    rules: list[Rule] = []
    configuration_id: str | None = None

class ParseConfig(ParseV2Parameters):
    model_config = ConfigDict(extra="allow")
    configuration_id: str | None = None

class SplitConfig(SplitV1Parameters):
    categories: list[SplitCategory] = []
    configuration_id: str | None = None

class BlockVersion(BaseModel):
    schema_version: str
    config_version: str

class PipelineVersions(BaseModel):
    classify: BlockVersion
    extract: BlockVersion
    split: BlockVersion

class SplitUploadSlot(BaseModel):
    id: str
    label: str
    parts: list[str]
    required: bool = False

class SplitUploadType(BaseModel):
    label: str
    slots: list[SplitUploadSlot] = Field(default_factory=list)
    extract_field_sources: dict[str, Any] | None = None

class SplitUploadConfig(BaseModel):
    extract_field_sources: dict[str, Any] = Field(default_factory=dict)
    types: dict[str, SplitUploadType] = Field(default_factory=dict)

class Config(BaseModel):
    """Root configuration model for configs/config.json."""
    config_id: str
    schema_version: str
    config_version: str
    pipeline_versions: PipelineVersions
    classify: ClassifyConfig
    extract_jubeex: ExtractConfig = Field(alias="extract-jubeex")
    parse: ParseConfig | None = None
    split: SplitConfig | None = None
    split_upload: SplitUploadConfig | None = None


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.json"
_API_EXCLUDE = {
    "configuration_id",
    "product_type",
    "schema_version",
    "config_version",
}


def _block_versions(block: Any) -> dict[str, str]:
    if isinstance(block, Mapping):
        schema = block.get("schema_version")
        version = block.get("config_version")
    else:
        schema = getattr(block, "schema_version", None)
        version = getattr(block, "config_version", None)
    out: dict[str, str] = {}
    if schema:
        out["schema_version"] = str(schema)
    if version:
        out["config_version"] = str(version)
    return out


@lru_cache(maxsize=1)
def load_config_payload() -> dict[str, Any]:
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def config_identity(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Version stamp included in metadata, artifacts, and stored extract JSON."""
    data = payload if payload is not None else load_config_payload()
    versions = data.get("pipeline_versions")
    version_map = versions if isinstance(versions, Mapping) else {}
    return {
        "config_id": data.get("config_id"),
        "schema_version": data.get("schema_version"),
        "config_version": data.get("config_version"),
        "classify": _block_versions(
            version_map.get("classify") or data.get("classify")
        ),
        "extract": _block_versions(
            version_map.get("extract") or data.get("extract-jubeex")
        ),
        "split": _block_versions(version_map.get("split") or data.get("split")),
    }


def dump_api_configuration(config: BaseModel) -> dict[str, Any]:
    """JSON sent to LlamaCloud classify / extract / split.

    Only LlamaCloud's own fields are forwarded. Version keys stay on our
    metadata and S3 artifacts.
    """
    dumped = config.model_dump(exclude_none=True)
    for key in _API_EXCLUDE:
        dumped.pop(key, None)
    parent = next(
        (
            base
            for base in type(config).__mro__[1:]
            if base
            in (
                ClassifyV2Parameters,
                ExtractV2Parameters,
                SplitV1Parameters,
                ParseV2Parameters,
            )
        ),
        None,
    )
    if parent is not None:
        allowed = set(parent.model_fields) - {"product_type"}
        dumped = {key: value for key, value in dumped.items() if key in allowed}
    return dumped


def with_config_identity(payload: dict[str, Any]) -> dict[str, Any]:
    stamped = dict(payload)
    stamped["config"] = config_identity()
    return stamped
