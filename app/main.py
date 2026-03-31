import io
import os
import tempfile
import time
from contextlib import asynccontextmanager

import instructor
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# Marker OCR imports
from marker.convert import convert_single_pdf
from marker.models import load_all_models
from openai import OpenAI
from PIL import Image

from app.schema import CandidateExtraction

# Initialize using the Groq API key and base URL
client = instructor.from_openai(
    OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    ),
    mode=instructor.Mode.JSON,
)

# Global variable to hold ML models in memory
ml_models = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Runs once when the server starts.
    We load the heavy Marker OCR models into memory here so they
    are instantly available for incoming API requests.
    """
    global ml_models
    print("Loading Marker OCR models into memory...")
    ml_models = load_all_models()
    print("Models loaded successfully.")
    yield
    # Cleanup on shutdown
    ml_models = None


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
    temp_pdf_path = None

    try:
        # Create a temporary file placeholder for the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf_path = temp_pdf.name

        file_bytes = await file.read()

        # 2. Format Routing: PDF vs Image
        if filename.endswith(".pdf"):
            # It's already a PDF, just write the bytes to the temp file
            with open(temp_pdf_path, "wb") as f:
                f.write(file_bytes)
        else:
            # It's an image. Convert it to a PDF in memory.
            image = Image.open(io.BytesIO(file_bytes))

            # PDFs do not support RGBA (transparency in PNGs). Convert to standard RGB.
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Save the image directly into our temporary PDF path
            image.save(temp_pdf_path, "PDF", resolution=100.0)

        # 3. Proceed with Marker exactly as before
        full_text, images, out_meta = convert_single_pdf(temp_pdf_path, ml_models)

        if not full_text or len(full_text.strip()) < 20:
            raise HTTPException(
                status_code=422,
                detail="Could not extract readable text from the document.",
            )

        system_prompt = """
        You are an expert ATS data extraction system tailored for the Indian IT job market.
        Extract the requested fields from the resume markdown.
        Pay special attention to calculating total experience accurately into Years and Months.
        If any of the requested fields are not explicitly stated in the resume, return null for those fields rather than guessing them.
        Standardize the highest education qualification to common acronyms (e.g., B-TECH, M-TECH, MCA).
        """

        candidate_data = client.chat.completions.create(
            model="gpt-4o-mini",  # Or Llama-3 on Groq
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
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
