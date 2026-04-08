from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SkillExtraction(BaseModel):
    # The LLM cannot know your database's numeric skillId.
    # The backend must map this extracted string to the correct DB skillId.
    skillName: str = Field(
        description="The exact name of the technical skill, tool, or framework (e.g., 'AWS Deployment', 'Spring Boot')."
    )
    skillLevel: Literal["Beginner", "Intermediate", "Advanced"] = Field(
        description="Estimate the proficiency level based on years of experience or context. Default to 'Intermediate' if unsure."
    )


class JobRecord(BaseModel):
    companyName: str = Field(description="The name of the company or employer.")
    bulletPoints: List[str] = Field(
        description="Extract EVERY SINGLE bullet point and sentence under this job exactly as written. Do not summarize. Create a new string in this array for each bullet point.",
        default_factory=list,
    )


class ProfessionalDetails(BaseModel):
    professionalSummary: Optional[str] = Field(
        description="""The candidate's professional summary, profile, or objective statement.
        Extract this word-for-word if present. Do not summarize.""",
        default=None,
    )
    workExperienceDetails: List[JobRecord] = Field(
        description="An array containing a record for EVERY job listed in the Experience/Employment history section. Do not skip any jobs.",
        default_factory=list,
    )
    projectDetails: Optional[str] = Field(
        description="""A combined string of the descriptions strictly from a STANDALONE 'Projects' section.
        CRITICAL: If a project or client is listed underneath an employer in the Work Experience section, DO NOT put it here. Leave it in the workExperienceDetails.""",
        default=None,
    )
    educationAndCertifications: Optional[str] = Field(
        description="""Details regarding the candidate's degrees, universities, and certifications.
        Extract word-for-word.""",
        default=None,
    )


class CandidateExtraction(BaseModel):
    """
    Schema for extracting candidate information to perfectly match the HRMS frontend CandidateFormValues interface.
    """

    # PHASE 1: Personal Identity & Contact
    firstName: str = Field(description="Candidate's first name. Convert to title case")
    middleName: Optional[str] = Field(
        description="Candidate's middle name, if available. Convert to title case",
        default=None,
    )
    lastName: str = Field(
        description="Candidate's last name. It must be a single word. Do not include the middle name or initial here. Convert to title case."
    )
    gender: Optional[Literal["Male", "Female", "Other"]] = Field(
        description="Infer gender from name or pronouns if possible",
        default=None,
    )
    emailId: Optional[str] = Field(
        description="Candidate's email address", default=None
    )
    contactNumber: Optional[str] = Field(
        description="Primary phone number (10-15 digits, include the country code if present. If not present use '+91' as the default country code)",
        default=None,
    )
    alternateNumber: Optional[str] = Field(
        description="Secondary phone number (10-15 digits), if available", default=None
    )
    dateOfBirth: Optional[str] = Field(
        description="Date of birth explicitly formatted exactly as YYYY-MM-DD for JavaScript Date compatibility.",
        default=None,
    )

    # Location Details
    presentAddress: Optional[str] = Field(
        description="The complete present street address", default=None
    )
    currentLocation: Optional[str] = Field(
        description="Current city name (e.g., 'Bangalore', 'Chennai')", default=None
    )
    preferredLocation: Optional[str] = Field(
        description="Preferred relocation city, if mentioned", default=None
    )
    willingToRelocate: Optional[bool] = Field(
        description="True if the candidate explicitly mentions willingness to relocate",
        default=None,
    )
    pincode: Optional[str] = Field(
        description="6-digit postal code/pincode if present", default=None
    )

    # System IDs (LLM will default these to None, backend/frontend must handle them)
    jdId: Optional[str] = Field(
        description="Job Description ID. Always return null.", default=None
    )
    sourceId: Optional[str] = Field(
        description="Source ID. Always return null.", default=None
    )

    # PHASE 2: Professional Background
    presentCompany: Optional[str] = Field(
        description="Current or most recent company name from the professional experience or employment history sections",
        default=None,
    )
    jobRole: Optional[str] = Field(
        description="Current or most recent Job Title / Role", default=None
    )
    educationQualification: Optional[str] = Field(
        description="Standardized highest degree (e.g., 'B-TECH', 'M-TECH', 'BCA', 'MCA', 'B.Sc')",
        default=None,
    )

    # Combined Experience Fields (Frontend expects strings)
    experienceYears: Optional[str] = Field(
        description="Total professional experience represented as a string (e.g., '5.5' for 5 and a half years)",
        default=None,
    )
    relevantExperience: Optional[str] = Field(
        description="Experience relevant to core technical skills as a string (e.g., '3.0')",
        default=None,
    )

    # Compensation & Availability
    noticePeriodDays: Optional[str] = Field(
        description="Notice period in days as a string (e.g., '30', '60').",
        default=None,
    )
    fixedSalaryLpa: Optional[str] = Field(
        description="Current Fixed Salary in LPA (Lakhs Per Annum) as a string up to 3 decimal places (e.g., '12.500').",
        default=None,
    )
    isVariableSalary: Optional[bool] = Field(
        description="Set to true if a variable salary or bonus component is mentioned.",
        default=False,
    )
    variableSalaryLpa: Optional[str] = Field(
        description="Variable Salary in LPA as a string, if mentioned.",
        default=None,
    )
    expectedCtc: Optional[str] = Field(
        description="Expected CTC in LPA as a string up to 3 decimal places.",
        default=None,
    )

    employmentType: Optional[str] = Field(
        description="Type of employment (e.g., 'Full-Time', 'Contract')", default=None
    )
    referredById: Optional[str] = Field(
        description="Referrer ID. Always return null.", default=None
    )
    isReferred: Optional[bool] = Field(
        description="Always return false.", default=False
    )

    # Technical Expertise
    skills: List[SkillExtraction] = Field(
        description="List of extracted technical skills mapped to a competency level.",
        default_factory=list,
    )

    # Professional detail data for candidate ranking engine
    professionalDetails: Optional[ProfessionalDetails] = Field(
        description="Structured extraction of the candidate's unstructured professional text.",
        default=None,
    )
