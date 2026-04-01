import os
import tempfile
import time
from contextlib import asynccontextmanager

import fitz
import instructor
from fastapi import FastAPI, File, HTTPException, UploadFile

# Marker OCR imports
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from openai import OpenAI

from app.schema import CandidateExtraction

# Initialize using the Groq API key and base URL
client = instructor.from_openai(
    OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    ),
    mode=instructor.Mode.JSON,
)

# Global variable to hold converter and models in memory
converter_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Runs once when the server starts.
    We load the heavy Marker OCR models into memory here so they
    are instantly available for incoming API requests.
    """
    global converter_instance
    print("Loading Marker OCR models into memory...")
    models_dict = create_model_dict()
    converter_instance = PdfConverter(artifact_dict=models_dict)
    print("Models loaded successfully.")
    yield
    # Cleanup on shutdown
    converter_instance = None


app = FastAPI(title="Resume Auto-Populate API", lifespan=lifespan)


@app.post("/api/v1/extract-resume")
async def extract_resume(file: UploadFile = File(...)):
    # 1. Expand allowed extensions
    allowed_extensions = (".pdf", ".png", ".jpg", ".jpeg")
    filename = file.filename.lower()

    if not filename.endswith(allowed_extensions):
        raise HTTPException(
            status_code=400, detail="Only PDF, PNG, and JPG files are supported."
        )

    start_time = time.time()
    tmp_file_path = None

    try:
        # 2. Create temporary file for the pdf since the conberter requires it
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_file_path = tmp.name

        # Try to parse the file using PyMuPDF (fitz)
        full_text = ""
        needs_ocr = False

        if suffix == ".pdf":
            try:
                doc = fitz.open(tmp_file_path)
                for page in doc:
                    full_text += page.get_text()
                doc.close()
                # If the PDF is just a scanned image, it will yield very little text
                if len(full_text.strip()) < 50:
                    print(
                        "Parsing using PyMuPDF yielded very little text, triggering Marker OCR"
                    )
                    needs_ocr = True
            except Exception as e:
                print(f"An error occurred while parsing the PDF using PyMuPDF:\n{e}")
                needs_ocr = True
        else:
            print("Suffix is not a pdf")
            needs_ocr = True

        # If the file is not a PDF or the PDF is a scanned image, use Marker OCR
        if needs_ocr:
            # This handles PDFs and Images natively
            print(f"Triggering Marker OCR for {filename}...")
            rendered = converter_instance(tmp_file_path)
            full_text, _, _ = text_from_rendered(rendered)

        if not full_text or len(full_text) < 20:
            raise HTTPException(
                status_code=422, detail="Extraction using Marker Converter failed"
            )

        # 3. Pass the retrieved text from the pdf to an LLM for semantic retrieval with a particular schema
        system_prompt = """
        You are an expert ATS data extraction system tailored for the Indian IT job market.
        Extract the requested fields from the resume markdown.
        Pay special attention to calculating total experience accurately into Years and Months.
        Current/Expected CTC should be extracted as floats representing Lakhs Per Annum (LPA).
        Standardize the highest education qualification to common acronyms (e.g., B-TECH, M-TECH, MCA).
        If any of the requested fields are not explicitly stated in the resume, return null for those fields rather than guessing them.
        """

        candidate_data = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_model=CandidateExtraction,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Resume Markdown:\n\n{full_text}"},
            ],
            temperature=0.0,
        )

        processing_time = round((time.time() - start_time) * 1000)

        return {
            "status": "success",
            "processing_time_ms": processing_time,
            "data": candidate_data.model_dump(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 4. Clean up the temporary file regardless of success or failure
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
