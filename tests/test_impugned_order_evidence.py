from extraction_review.extract_record import apply_extract_envelope


def test_narrative_order_date_rejected_even_in_impugned_slot():
    record = {
        "impugned_orders": [
            {
                "is_primary": True,
                "order_date": "4.5.21",
                "source_part": "Impugned Order",
                "source_pages": [4],
                "raw_text": "this Hon'ble Court in its order dated 4.5.21 which is summarized herein as follows:-",
            }
        ]
    }
    result = apply_extract_envelope(record)
    assert result["impugned_orders"] == []
    assert record["impugned_orders"][0]["order_date"] == "4.5.21"


def test_explicit_challenged_order_and_standalone_order_dates_retained():
    orders = [
        {
            "order_date": "4.5.21",
            "source_part": "Main Petition",
            "raw_text": "Against the impugned judgment dated 4.5.21, summarized as follows.",
        },
        {
            "order_date": "4.5.21",
            "source_part": "Impugned Order",
            "raw_text": "IN THE HIGH COURT\nWrit Petition No. 123/2021\nJUDGMENT\nDate: 4.5.21",
        },
    ]
    assert (
        apply_extract_envelope({"impugned_orders": orders})["impugned_orders"] == orders
    )


def test_remove_only_reference_record_from_multiple_orders():
    orders = [
        {
            "order_date": "4.5.21",
            "raw_text": "In compliance with the order dated 4.5.21, prisoners were released.",
        },
        {
            "order_date": "8.6.22",
            "raw_text": "The impugned order dated 8.6.22 is challenged in this petition.",
        },
    ]
    assert (
        apply_extract_envelope({"impugned_orders": orders})["impugned_orders"]
        == orders[1:]
    )
