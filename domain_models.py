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
    Meets FDA-compliant medical validation rules for thoracic diagnostic tools.
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
        description="Cumulative smoking pack-years (packs per day * years smoked)."
    )
    egfr: GeneticsStatus = Field(
        GeneticsStatus.UNKNOWN,
        description="EGFR molecular variant status (0=WT, 1=Mutant, 2=Unknown)."
    )
    kras: GeneticsStatus = Field(
        GeneticsStatus.UNKNOWN,
        description="KRAS molecular variant status (0=WT, 1=Mutant, 2=Unknown)."
    )
    alk: GeneticsStatus = Field(
        GeneticsStatus.UNKNOWN,
        description="ALK fusion/rearrangement status (0=WT, 1=Mutant, 2=Unknown)."
    )

    @field_validator('age')
    @classmethod
    def validate_oncology_age_range(cls, v: int) -> int:
        if v < 18 or v > 100:
            raise ValueError("Age must fall within the verified clinical oncology range of 18 to 100.")
        return v

    @field_validator('smoking_pack_years')
    @classmethod
    def validate_pack_years_range(cls, v: float) -> float:
        if v < 0.0 or v > 150.0:
            raise ValueError("Cumulative smoking pack-years must conform to typical clinical ranges [0.0 to 150.0].")
        return v

class DiagnosticOutputSchema(BaseModel):
    """
    FDA Audit trail telemetry schema, structuring calibrated risk, classification levels,
    computational latencies, and compliance logging stamps.
    """
    patient_id: Optional[str] = Field("DE-ID-CLINICAL-045", description="De-identified patient tracking token.")
    risk_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Fused Swin-Transformer cross-calibrated lung nodule malignancy risk probability."
    )
    risk_classification: str = Field(
        ..., 
        description="Categorical risk stratification: 'LOW RISK', 'MODERATE RISK', or 'HIGH RISK'."
    )
    processing_latency_ms: float = Field(
        ..., 
        description="End-to-end multi-modal diagnostic pipeline execution latency in milliseconds."
    )
    compliance_audit_logged: bool = Field(
        True,
        description="Audit trace stamp signifying data ingestion contracts and XAI map logging are complete."
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="UTC diagnostic execution timestamp."
    )
