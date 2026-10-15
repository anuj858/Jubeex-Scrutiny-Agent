"""First-30-page extract: slice, schema, and a null special category."""

from __future__ import annotations

from types import SimpleNamespace

import pymupdf
import pytest

from extraction_review.config import ExtractConfig, FrontExtractRecord, LegalExtractRecord
from extraction_review.extract_record import finalize_front_extract
from extraction_review.process_file import (
    FRONT_EXTRACT_END_PAGE_LIMIT,
    FRONT_EXTRACT_OPENING_CAP,
    FRONT_EXTRACT_PAGE_LIMIT,
    FileEvent,
    first_pages_pdf,
    front_extract_opening_pages,
)
from extraction_review.split_upload import (
    build_extract_pack_markdown,
    build_front_extract_pack_markdown,
    build_front_extract_system_prompt,
    front_extract_configuration,
)


def _pdf_bytes(page_count: int) -> bytes:
    document = pymupdf.open()
    try:
        for _ in range(page_count):
            document.new_page()
        return document.tobytes()
    finally:
        document.close()


def _page_count(pdf_bytes: bytes) -> int:
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return document.page_count
    finally:
        document.close()


def _marked_pdf(page_count: int) -> bytes:
    document = pymupdf.open()
    try:
        for number in range(1, page_count + 1):
            page = document.new_page()
            page.insert_text((72, 72), f"PAGE-{number}")
        return document.tobytes()
    finally:
        document.close()


def _page_texts(pdf_bytes: bytes) -> list[str]:
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return [page.get_text() for page in document]
    finally:
        document.close()


def test_front_slice_keeps_the_opening_and_the_last_five_pages() -> None:
    sliced = first_pages_pdf(_marked_pdf(40))
    texts = _page_texts(sliced)
    assert len(texts) == FRONT_EXTRACT_PAGE_LIMIT + FRONT_EXTRACT_END_PAGE_LIMIT
    joined = "\n".join(texts)
    assert "PAGE-1" in joined
    assert "PAGE-30" in joined
    assert "PAGE-31" not in joined
    assert "PAGE-36" in joined
    assert "PAGE-40" in joined


def test_front_slice_does_not_repeat_pages_already_in_the_opening() -> None:
    sliced = first_pages_pdf(_marked_pdf(32))
    assert _page_count(sliced) == 32


def test_opening_stays_at_30_until_200_then_shrinks_below_20_percent() -> None:
    assert front_extract_opening_pages(100) == 30
    assert front_extract_opening_pages(200) == 30
    assert front_extract_opening_pages(700) == 75
    assert front_extract_opening_pages(700) >= 34
    assert front_extract_opening_pages(1000) == 90
    assert front_extract_opening_pages(1000) < 200
    assert front_extract_opening_pages(5000) == FRONT_EXTRACT_OPENING_CAP


def test_long_book_keeps_the_main_petition_start_and_the_vakalatnama() -> None:
    opening = front_extract_opening_pages(700)
    sliced = first_pages_pdf(_marked_pdf(700), opening)
    joined = "\n".join(_page_texts(sliced))
    assert "PAGE-34" in joined
    assert "PAGE-76" not in joined
    assert "PAGE-696" in joined
    assert "PAGE-700" in joined
    assert _page_count(sliced) == opening + FRONT_EXTRACT_END_PAGE_LIMIT


def test_first_pages_pdf_keeps_a_shorter_book() -> None:
    raw = _pdf_bytes(12)
    assert first_pages_pdf(raw) == raw
    assert _page_count(first_pages_pdf(raw)) == 12


def test_special_category_sits_beside_petition_type_on_the_front_schema_only() -> None:
    legal = list(LegalExtractRecord.model_json_schema()["properties"])
    assert "special_category" not in legal
    front = front_extract_configuration(
        ExtractConfig.model_construct(data_schema={}, product_type="extract")
    )
    props = list(front["data_schema"]["properties"])
    assert props.index("special_category") == props.index("petition_type") + 1
    assert "special_category" in FrontExtractRecord.model_fields
    assert "special_category" not in LegalExtractRecord.model_fields
    prompt = build_front_extract_system_prompt()
    assert "Vakalatnama" in prompt
    assert "List of Dates" in prompt
    assert "special_category" in prompt
    assert "configuration_id" not in front


def test_front_pack_keeps_the_middle_of_the_30_page_slice() -> None:
    pages = {page: f"page {page} text" for page in range(1, 36)}
    labels = {page: ["Main Petition"] for page in range(1, 31)}
    normal = build_extract_pack_markdown(pages, labels, {"Main Petition"})
    front = build_front_extract_pack_markdown(pages)
    assert "page 20 text" not in normal
    assert "page 1 text" in front
    assert "page 20 text" in front
    assert "page 30 text" in front
    assert "## [Vakalatnama] (p. 31)" in front
    assert "page 35 text" in front
    wide = build_front_extract_pack_markdown(pages, opening_pages=75)
    assert "## [Compiled opening] (p. 31)" in wide
    assert "## [Vakalatnama] (p. 31)" not in wide


def test_missing_special_category_stays_null_and_later_pages_stay_empty() -> None:
    payload = {
        "court": "Supreme Court of India",
        "petition_type": "Special Leave Petition (Civil)",
        "cause_title": {"formatted_title": "A VS B"},
        "special_category": "N/A",
        "petitioners": [],
        "respondents": [],
        "advocates_on_record": [{"name": "Ravi"}],
        "impugned_orders": [],
        "relief_sort": "Grant leave to appeal",
        "petition_date": "01.01.2024",
        "documents": [{"name": "Main Petition"}],
        "document_counts": {"processed": 1, "failed": 0},
    }
    done = finalize_front_extract(payload)
    keys = list(done)
    assert keys.index("special_category") == keys.index("petition_type") + 1
    assert done["special_category"] is None
    assert done["petition_type"] == "Special Leave Petition (Civil)"
    assert done["relief_sort"] is None
    assert done["petition_date"] is None
    assert done["documents"] == []
    assert done["document_counts"] == {"processed": 0, "failed": 0}
    assert done["advocates_on_record"] == [{"name": "Ravi"}]


_MEMO = """
MEMO OF PARTIES
IN THE MATTER OF:-
...Petitioner
Versus
...Respondents
Vinod Kumar Mittal
S/o Late Sh. Kanhaiya Lal Mittal
R/o -869, First Floor, New Rajinder
Nagar, Delhi-110060.
1. State NCT of Delhi
Through SHO PS Economic Offences Wing, New Delhi-110001.
2. Pawan Sachdeva
"""


def test_front_parties_come_from_the_memo_not_the_checklist() -> None:
    payload = {
        "petition_type": "SPECIAL LEAVE PETITION (CRL.)",
        "cause_title": {
            "formatted_title": "VINOD KUMAR MITTAL and Ors. VS STATE NCT OF DELHI and Ors."
        },
        "petitioners": [
            {"name": "1. VINOD KUMAR MITTAL"},
            {
                "name": (
                    "The Petition is barred by time and there is delay of days "
                    "in filing the same"
                )
            },
            {"name": "(a). Main Category classification: 14"},
            {"name": "Criminal Matters: Yes"},
            {"name": "Special Category (First Petitioner/appellant only): NA"},
        ],
        "respondents": [
            {"name": "STATE NCT OF DELHI"},
            {"name": "Learned counsel appearing on behalf of the petitioner submits that"},
            {"name": "The petition is dismissed and disposed of"},
            {"name": "Pending applications, if any, also stand disposed of"},
        ],
    }
    done = finalize_front_extract(payload, page_markdown={338: _MEMO})
    assert [party["name"] for party in done["petitioners"]] == ["Vinod Kumar Mittal"]
    assert [party["name"] for party in done["respondents"]] == [
        "State NCT of Delhi",
        "Pawan Sachdeva",
    ]
    assert done["cause_title"]["formatted_title"] == (
        "Vinod Kumar Mittal VS State NCT of Delhi and Anr."
    )


def test_printed_special_category_is_kept() -> None:
    done = finalize_front_extract(
        {
            "petition_type": "Writ Petition (Civil)",
            "special_category": "Tax Matters",
        }
    )
    assert done["special_category"] == "Tax Matters"


@pytest.mark.asyncio
async def test_extract_front_uploads_the_30_page_slice_and_skips_the_request_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extraction_review.process_file import _run_extract_front

    uploaded: dict[str, bytes] = {}
    captured: dict[str, object] = {}

    async def fake_load(*_args: object, **_kwargs: object) -> bytes:
        return _pdf_bytes(45)

    async def fake_upload(
        _client: object,
        *,
        filename: str,
        pdf_bytes: bytes,
        external_file_id: str | None = None,
    ) -> str:
        del filename, external_file_id
        uploaded["pdf"] = pdf_bytes
        return "dfl-front-slice"

    async def fake_extract(_ctx: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "agd-front"

    monkeypatch.setattr(
        "extraction_review.process_file._load_bundle_pdf",
        fake_load,
    )
    monkeypatch.setattr(
        "extraction_review.process_file._upload_slot_pdf",
        fake_upload,
    )
    monkeypatch.setattr(
        "extraction_review.process_file._extract_sliced_parts",
        fake_extract,
    )
    event = FileEvent(
        job_type="extract_front",
        filing_type="SLP_CIVIL",
        special_category="Tax Matters",
        documents=[
            {
                "name": "compiled.pdf",
                "file_id": "dfl-full",
                "download_url": "https://example.com/compiled.pdf",
            }
        ],
    )
    result = await _run_extract_front(
        event,
        ctx=SimpleNamespace(write_event_to_stream=lambda _event: None),
        client=object(),
    )
    assert _page_count(uploaded["pdf"]) == (
        FRONT_EXTRACT_PAGE_LIMIT + FRONT_EXTRACT_END_PAGE_LIMIT
    )
    assert captured["front_extract"] is True
    assert captured["skip_index"] is True
    assert captured["special_category"] is None
    assert captured["court"] is None
    parts = captured["parts"]
    assert parts[0].file_id == "dfl-front-slice"
    assert result.agent_data_id == "agd-front"
    assert result.parsed_slots == ["fullpetition"]
    assert len(event.documents) == 1
    assert result.job_type == "extract_front"
