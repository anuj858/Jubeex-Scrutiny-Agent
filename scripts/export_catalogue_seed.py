"""Write sci_defect_seed.v1.json from the JSON defect catalogue.

uv run python scripts/export_catalogue_seed.py --out /tmp/sci_defect_seed.v1.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from extraction_review.scrutiny.catalogue_export import write_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination JSON path",
    )
    args = parser.parse_args()
    payload = write_seed(args.out)
    print(
        f"Wrote {args.out} "
        f"({len(payload['defects'])} defects, {len(payload['sources'])} sources)"
    )


if __name__ == "__main__":
    main()
