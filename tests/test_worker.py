from unittest.mock import patch

import pytest

from extraction_review.worker import run_worker


@pytest.mark.asyncio
async def test_worker_requires_tesseract_runtime() -> None:
    with (
        patch("extraction_review.worker.sqs_enabled", return_value=True),
        patch("extraction_review.worker.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="install tesseract-ocr"),
    ):
        await run_worker(once=True)
