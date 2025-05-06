from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any


class SurveillanceMetadata(BaseModel):
    id: int = Field(..., description="Unique OSM node ID")
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")

    camera_type: Optional[str] = Field(None, description="e.g., 'dome', 'fixed', etc.")
    mount_type: Optional[str] = Field(None, description="e.g., 'wall', 'pole', etc.")
    zone: Optional[str] = Field(None, description="e.g., 'town', 'building', etc.")
    operator: Optional[str] = Field(None, description="Entity operating the camera")
    manufacturer: Optional[str] = Field(None, description="Manufacturer of the camera")
    public: Optional[bool] = Field(
        None, description="True if public surveillance, False if private"
    )
    surveillance_type: Optional[str] = Field(
        None, description="e.g., 'camera', 'guard', etc."
    )
    start_date: Optional[str] = Field(
        None, description="ISO date string (e.g., '2024-12-01')"
    )
    sensitive: bool = Field(
        False, description="Flagged as sensitive (e.g., police, government)"
    )

    original_tags: Dict[str, Any] = Field(
        ..., description="Original OSM tag dictionary"
    )
    extra_fields: Dict[str, Any] = Field(
        default_factory=dict, description="Any additional tags not explicitly modeled"
    )

    @field_validator("start_date")
    @classmethod
    def validate_start_date(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if value.endswith("?"):
            return value  # Allow fuzzy date like "2024?"
        # Ensure it’s a valid partial or full ISO date
        import re

        if re.match(r"^\d{4}(-\d{2}-\d{2})?$", value):
            return value
        raise ValueError("start_date must be in format YYYY or YYYY-MM-DD or YYYY?")

    @classmethod
    def from_raw(cls, element: dict, enriched_fields: dict) -> "SurveillanceMetadata":
        """
        Combines the raw OSM element and LLM-enriched metadata into a validated object.
        """
        tags = element.get("tags", {})
        known_fields = {
            "camera_type",
            "mount_type",
            "zone",
            "operator",
            "manufacturer",
            "public",
            "surveillance_type",
            "start_date",
            "sensitive",
        }
        return cls(
            id=element["id"],
            lat=element["lat"],
            lon=element["lon"],
            camera_type=enriched_fields.get("camera_type"),
            mount_type=enriched_fields.get("mount_type"),
            zone=enriched_fields.get("zone"),
            operator=enriched_fields.get("operator"),
            manufacturer=enriched_fields.get("manufacturer"),
            public=enriched_fields.get("public"),
            surveillance_type=enriched_fields.get("surveillance_type"),
            start_date=enriched_fields.get("start_date"),
            sensitive=enriched_fields.get("sensitive", False),
            original_tags=tags,
            extra_fields={k: v for k, v in tags.items() if k not in known_fields},
        )
