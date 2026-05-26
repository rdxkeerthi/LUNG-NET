from pydantic import BaseModel, Field, field_validator
from enum import IntEnum
from datetime import datetime
from typing import Optional

class GeneticsStatus(IntEnum):
    """
    Oncological molecular genetic marker statuses:
    0: Wild-Type (WT) / Negative
    1: Mutant (MUT) / Positive
    2: Unknown / Untested
    """
    WT = 0
    MUTANT = 1
    UNKNOWN = 2

class PatientClinicalPayload(BaseModel):
    """
    Patient Clinical & Demographics Data Profile with strict Pydantic V2 bounds.
    """
    age: int = Field(
        ..., 
        ge=18, 
        le=100, 
        description="Patient age in years at date of diagnostic assessment."
    )
    smoking_pack_years: float = Field(
        ..., 
        ge=0.0, 
        le=150.0, 
        description="Cumulative smoking pack-years."
    )
    egfr: GeneticsStatus = Field(
        GeneticsStatus.UNKNOWN,
        description="EGFR molecular status."
    )
    kras: GeneticsStatus = Field(
        GeneticsStatus.UNKNOWN,
        description="KRAS molecular status."
    )
    alk: GeneticsStatus = Field(
        GeneticsStatus.UNKNOWN,
        description="ALK fusion status."
    )

    @field_validator('age')
    @classmethod
    def validate_oncology_age_range(cls, v: int) -> int:
        if v < 18 or v > 100:
            raise ValueError("Age must fall within the clinical oncology range of 18 to 100.")
        return v

    @field_validator('smoking_pack_years')
    @classmethod
    def validate_pack_years_range(cls, v: float) -> float:
        if v < 0.0 or v > 150.0:
            raise ValueError("Smoking pack-years must fall within valid bounds [0.0 to 150.0].")
        return v

class DiagnosticOutputSchema(BaseModel):
    """
    FDA Audit trail telemetry schema.
    """
    patient_id: Optional[str] = Field("DE-ID-CLINICAL-045", description="De-identified tracking token.")
    risk_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Calibrated lung nodule malignancy risk probability."
    )
    risk_classification: str = Field(
        ..., 
        description="Stratification category: 'LOW RISK', 'MODERATE RISK', or 'HIGH RISK'."
    )
    processing_latency_ms: float = Field(
        ..., 
        description="Inference pipeline execution latency in milliseconds."
    )
    compliance_audit_logged: bool = Field(
        True,
        description="Audit trace stamp."
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="UTC timestamp."
    )
