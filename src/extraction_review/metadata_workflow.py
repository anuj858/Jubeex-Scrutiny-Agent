import logging
from typing import Annotated, Any

from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig

from .config import (
    EXTRACTED_DATA_COLLECTION,
    JUBEEX_FILING_TYPES,
    CoreFilingRecord,
    ExtractConfig,
)
from .split_upload import ui_catalog

logger = logging.getLogger(__name__)

DISCRIMINATOR_FIELD = "petition_type"


class MetadataResponse(StopEvent):
    json_schema: dict[str, Any]
    schemas: dict[str, dict[str, Any]]
    discriminator_field: str
    extracted_data_collection: str
    split_upload_types: dict[str, Any]


async def get_presentation_schema(
    extract_jubeex: Annotated[
        ExtractConfig,
        ResourceConfig(
            config_file="configs/config.json",
            path_selector="extract-jubeex",
            label="JubeeX Extraction",
        ),
    ],
) -> dict[str, Any]:
    del extract_jubeex
    schema = CoreFilingRecord.model_json_schema()
    schemas = {ftype: schema for ftype in JUBEEX_FILING_TYPES}
    return {
        "json_schema": schema,
        "schemas": schemas,
        "discriminator_field": DISCRIMINATOR_FIELD,
    }


class MetadataWorkflow(Workflow):
    """Provide extraction schema and configuration to the workflow editor."""

    @step
    async def get_metadata(
        self,
        _: StartEvent,
        presentation: Annotated[dict[str, Any], Resource(get_presentation_schema)],
    ) -> MetadataResponse:
        """Return the data schemas and storage settings for the review interface."""
        logger.info("[Metadata] Serving extraction schema to the UI")
        return MetadataResponse(
            json_schema=presentation["json_schema"],
            schemas=presentation["schemas"],
            discriminator_field=presentation["discriminator_field"],
            extracted_data_collection=EXTRACTED_DATA_COLLECTION,
            split_upload_types=ui_catalog(),
        )


workflow = MetadataWorkflow(timeout=None)
