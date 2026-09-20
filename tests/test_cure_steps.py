from extraction_review.scrutiny.prompts import display_cure_steps, _cure_aim


def test_display_cure_steps_keeps_catalogue_substance() -> None:
    steps = [
        "1. Prompt the User to provide the complete residential and/or office addresses of all the parties, and auto-fill the same in the Petition/I.A. with Jubeex.",
        "2. Option for the User to edit the addresses of the parties with Jubeex.",
        "3. Re-upload the Petition/I.A.",
    ]
    shown = display_cure_steps(steps)
    assert shown[0].startswith("1. Prompt the User to provide")
    assert "Include 1." not in shown
    assert "addresses" in shown[0]
    assert shown[2].startswith("3. Re-upload")


def test_cure_aim_does_not_collapse_to_include_number() -> None:
    aim = _cure_aim(
        "1. Prompt the User to provide the complete residential and/or office "
        "addresses of all the parties, and auto-fill the same in the Petition/I.A. with Jubeex."
    )
    assert aim != "Include 1."
    assert "addresses" in aim.lower()
    assert not aim.endswith(".A.")
