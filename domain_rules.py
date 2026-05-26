from pydantic import BaseModel, Field, field_validator
from enum import IntEnum
from typing import Dict, Any

class GeneticsVariant(IntEnum):
    """
    Oncogene variant statuses:
    0: Wild-Type (WT) / Negative
    1: Mutant (MUT) / Positive
    2: Unknown / Untested
    """
    WT = 0
    MUTANT = 1
    UNKNOWN = 2


class ClinicalDataModel(BaseModel):
    """
    Pydantic V2 demographic validation schema enforcing strict oncology boundaries.
    """
    age: int = Field(
        ..., 
        ge=18, 
        le=100, 
        description="Patient age in years at date of screening."
    )
    smoking_pack_years: float = Field(
        ..., 
        ge=0.0, 
        le=150.0, 
        description="Cumulative smoking history in pack-years."
    )
    egfr: GeneticsVariant = Field(
        default=GeneticsVariant.UNKNOWN, 
        description="EGFR oncogene status."
    )
    kras: GeneticsVariant = Field(
        default=GeneticsVariant.UNKNOWN, 
        description="KRAS oncogene status."
    )
    alk: GeneticsVariant = Field(
        default=GeneticsVariant.UNKNOWN, 
        description="ALK fusion status."
    )

    @field_validator('age')
    @classmethod
    def validate_oncology_age(cls, v: int) -> int:
        if v < 18 or v > 100:
            raise ValueError("Age must fall within the designated clinical range of 18 to 100.")
        return v

    @field_validator('smoking_pack_years')
    @classmethod
    def validate_pack_years(cls, v: float) -> float:
        if v < 0.0 or v > 150.0:
            raise ValueError("Smoking pack-years must be bounded between 0.0 and 150.0.")
        return v


class ClinicalRecommendationEngine:
    """
    Clinical Decision Logic implementing Fleischner Society guidelines
    and targeted molecular guidance.
    """
    @staticmethod
    def generate_report(risk_score: float, model_data: ClinicalDataModel) -> Dict[str, Any]:
        # Determine risk tier
        if risk_score < 0.30:
            risk_tier = "LOW RISK"
        elif risk_score < 0.70:
            risk_tier = "MODERATE RISK"
        else:
            risk_tier = "HIGH RISK"

        # Generate Fleischner action protocols
        actions = []
        followup_time = ""
        
        if risk_tier == "LOW RISK":
            if model_data.smoking_pack_years < 10.0:
                followup_time = "12 months"
                actions.append("Routine low-dose CT (LDCT) surveillance in 12 months.")
                actions.append("Recommend healthy lifestyle counseling and risk monitoring.")
            else:
                followup_time = "6 to 12 months"
                actions.append("Recommend repeat non-contrast low-dose chest CT in 6-12 months.")
                actions.append("Advise active smoking cessation programs.")
                
        elif risk_tier == "MODERATE RISK":
            followup_time = "3 to 6 months"
            actions.append("Perform repeat chest CT scan in 3-6 months to assess volume doubling time.")
            actions.append("Consider contrast-enhanced PET-CT scan to evaluate SUV max metabolic activity.")
            actions.append("Recommend referral to pulmonology for outpatient evaluation.")
            
        else: # HIGH RISK
            followup_time = "Immediate"
            actions.append("Immediate multidisciplinary thoracic oncology tumor board referral.")
            actions.append("Schedule interventional pulmonology or radiology consult for CT-guided core needle biopsy.")
            actions.append("Recommend contrast-enhanced PET-CT and brain MRI for systemic staging workup.")

        # Check molecular/genetic indications
        targeted_therapies = []
        if model_data.egfr == GeneticsVariant.MUTANT:
            targeted_therapies.append("EGFR mutant positive status: Consider first-line EGFR Tyrosine Kinase Inhibitors (TKIs) such as Osimertinib.")
        if model_data.kras == GeneticsVariant.MUTANT:
            targeted_therapies.append("KRAS mutant positive status: Consider KRAS-G12C inhibitors such as Sotorasib if G12C variant is verified.")
        if model_data.alk == GeneticsVariant.MUTANT:
            targeted_therapies.append("ALK translocation positive status: Consider ALK inhibitors such as Alectinib or Brigatinib.")

        report_summary = (
            f"CLINICAL INTERPRETATION REPORT\n"
            f"====================================\n"
            f"Risk Classification: {risk_tier} ({risk_score*100:.2f}% calculated probability)\n"
            f"Patient Age: {model_data.age} years | Smoking History: {model_data.smoking_pack_years} pack-years\n"
            f"Recommended Surveillance Interval: {followup_time}\n\n"
            f"Actionable Medical Directives:\n"
            + "\n".join([f"  - {act}" for act in actions])
        )

        if targeted_therapies:
            report_summary += (
                "\n\nTargeted Molecular Therapeutics Alerts:\n"
                + "\n".join([f"  - {tx}" for tx in targeted_therapies])
            )
        else:
            report_summary += "\n\nNo targetable oncogene mutations detected. Standard systemic chemotherapies apply if pathology is positive."

        return {
            "risk_tier": risk_tier,
            "followup_interval": followup_time,
            "directives": actions,
            "molecular_alerts": targeted_therapies,
            "raw_text_report": report_summary
        }
