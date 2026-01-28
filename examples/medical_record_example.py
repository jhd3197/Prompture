"""
Medical Record Processing Example

This example shows how to use Prompture for extracting patient information from medical documents.
It demonstrates registration of medical-specific fields and structured extraction of patient data
including conditions, medications, and allergies.
"""

from typing import Optional

from pydantic import BaseModel

from prompture import extract_with_model, field_from_registry, register_field

# Medical-specific field definitions
register_field(
    "medical_conditions",
    {
        "type": "list",
        "description": "List of diagnosed medical conditions",
        "instructions": "Extract diagnosed conditions, symptoms, and medical issues",
        "default": [],
        "nullable": True,
    },
)

register_field(
    "medications",
    {
        "type": "list",
        "description": "Current medications and prescriptions",
        "instructions": "Extract medication names, dosages, and frequencies",
        "default": [],
        "nullable": True,
    },
)

register_field(
    "allergies",
    {
        "type": "list",
        "description": "Known allergies and adverse reactions",
        "instructions": "Extract all known allergies, food sensitivities, drug reactions",
        "default": [],
        "nullable": True,
    },
)

register_field(
    "date_of_birth",
    {
        "type": "str",
        "description": "Date of birth in ISO format",
        "instructions": "Extract date of birth as YYYY-MM-DD format",
        "default": "",
        "nullable": True,
    },
)


class PatientRecord(BaseModel):
    name: str = field_from_registry("name")
    age: int = field_from_registry("age")
    date_of_birth: Optional[str] = field_from_registry("date_of_birth")
    medical_conditions: Optional[list[str]] = field_from_registry("medical_conditions")
    medications: Optional[list[str]] = field_from_registry("medications")
    allergies: Optional[list[str]] = field_from_registry("allergies")


# Sample medical record
medical_text = """
Patient: Robert Martinez, DOB: 1975-03-15, Age: 49

MEDICAL HISTORY:
- Type 2 Diabetes diagnosed 2018
- Hypertension since 2020
- Mild sleep apnea

CURRENT MEDICATIONS:
- Metformin 500mg twice daily
- Lisinopril 10mg once daily
- CPAP therapy for sleep apnea

ALLERGIES:
- Penicillin (causes rash)
- Shellfish (severe reaction)
"""

# Extract patient data
patient = extract_with_model(PatientRecord, medical_text, "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b")

print(f"Patient: {patient.model.name}, Age: {patient.model.age}")
print(f"Conditions: {patient.model.medical_conditions}")
print(f"Medications: {patient.model.medications}")
print(f"Allergies: {patient.model.allergies}")
