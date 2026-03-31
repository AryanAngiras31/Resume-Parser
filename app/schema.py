from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SkillExtraction(BaseModel):
    name: str = Field(
        description="The exact name of the technical skill, tool, or framework (e.g., 'AWS Deployment', 'Spring Boot')."
    )
    competency: Literal["Beginner", "Intermediate", "Advanced"] = Field(
        description="Estimate the proficiency level based on years of experience or context in the resume. Default to 'Intermediate' if unsure."
    )


class CandidateExtraction(BaseModel):
    """
    Schema for extracting candidate information to perfectly match the HRMS frontend form.
    """

    # PHASE 1: Personal Identity & Contact
    first_name: str = Field(description="Candidate's first name")
    last_name: str = Field(description="Candidate's last name")
    middle_name: Optional[str] = Field(
        description="Candidate's middle name, if available", default=None
    )
    gender: Literal["Male", "Female", "Other", None] = Field(
        description="Infer gender from name or pronouns if possible, otherwise null",
        default=None,
    )
    date_of_birth: Optional[str] = Field(
        description="Date of birth in DD/MM/YYYY format, if explicitly stated",
        default=None,
    )
    email: str = Field(description="Candidate's email address", default=None)
    primary_contact: Optional[str] = Field(
        description="Primary phone number (do not include country code)",
        default=None,
    )
    alternate_contact: Optional[str] = Field(
        description="Secondary phone number, if available", default=None
    )

    # Location Details
    current_location: Optional[str] = Field(
        description="City name (e.g., 'Bangalore', 'Chennai')", default=None
    )
    pincode: Optional[str] = Field(
        description="6-digit postal code/pincode if present", default=None
    )
    full_present_address: Optional[str] = Field(
        description="The complete street address", default=None
    )

    # PHASE 2: Professional Background
    current_company: Optional[str] = Field(
        description="Current or most recent company name", default=None
    )
    designation: Optional[str] = Field(
        description="Current or most recent Job Title / Role", default=None
    )

    total_experience_years: int = Field(
        description="Total professional experience (Years component only)", default=0
    )
    total_experience_months: int = Field(
        description="Total professional experience (Remaining Months component, 0-11)",
        default=0,
    )

    relevant_experience_years: int = Field(
        description="Experience relevant to core technical skills (Years component)",
        default=0,
    )
    relevant_experience_months: int = Field(
        description="Experience relevant to core technical skills (Months component)",
        default=0,
    )

    # Education
    highest_qualification: Optional[str] = Field(
        description="Standardized highest degree (e.g., 'B-TECH', 'M-TECH', 'BCA', 'MCA', 'B.Sc')",
        default=None,
    )

    # Compensation & Availability
    notice_period_days: Optional[int] = Field(
        description="Notice period in days (e.g., 30, 60, 90). Extract if mentioned.",
        default=None,
    )
    current_salary_lpa: Optional[float] = Field(
        description="Current Salary in LPA (Lakhs Per Annum). Extract only the float value.",
        default=None,
    )
    expected_ctc_lpa: Optional[float] = Field(
        description="Expected CTC in LPA (Lakhs Per Annum). Extract only the float value.",
        default=None,
    )

    # Technical Expertise
    technical_expertise: List[SkillExtraction] = Field(
        description="List of extracted technical skills mapped to a competency level.",
        default_factory=list,
    )
