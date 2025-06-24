import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_bytes
import re

# Load model at startup
MODEL_PATH = "./hf_model"
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model = model.half()
tokenizer = processor.tokenizer

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_clauses(text):
    """Extract important clauses from contract text."""
    clauses = {
        "penalty": [],
        "termination": [],
        "payment": [],
        "confidentiality": [],
        "liability": []
    }
    
    # More robust regex to capture entire paragraphs/sections
    patterns = {
        "penalty": r"(?is)(penalty(?: clause)?.*?)(?:\n\s*\n|\Z)",
        "termination": r"(?is)(termination(?: clause)?.*?)(?:\n\s*\n|\Z)",
        "payment": r"(?is)(payment terms.*?)(?:\n\s*\n|\Z)",
        "confidentiality": r"(?is)(confidentiality.*?)(?:\n\s*\n|\Z)",
        "liability": r"(?is)(limitation of liability.*?)(?:\n\s*\n|\Z)"
    }
    
    for clause_type, pattern in patterns.items():
        matches = re.finditer(pattern, text)
        clauses[clause_type].extend(match.group(1).strip() for match in matches)
    
    return clauses

def analyze_risks(clauses):
    """Analyze clauses for potential risks."""
    risks = []
    
    # Example risk patterns
    risk_patterns = [
        (r"(?i)unlimited.*liability", "Unlimited liability clause detected"),
        (r"(?i)no.*refund", "No refund policy may be problematic"),
        (r"(?i)perpetual|forever", "Perpetual obligations detected"),
        (r"(?i)immediate.*termination", "Immediate termination clause present"),
        (r"(?i)shall.*pay.*damages", "Mandatory damages clause detected")
    ]
    
    for clause_list in clauses.values():
        for clause in clause_list:
            for pattern, risk_desc in risk_patterns:
                if re.search(pattern, clause):
                    risks.append({"description": risk_desc, "clause": clause})
    
    return risks

def run_ocr(image: Image.Image, prompt: str = "Read text in the image."):
    """Run OCR on the image with DOLPHIN model."""
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.half()
    prompt = f"<s>{prompt} <Answer/>"
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    decoder_attention_mask = torch.ones_like(prompt_ids)
    
    outputs = model.generate(
        pixel_values=pixel_values.to(device),
        decoder_input_ids=prompt_ids,
        decoder_attention_mask=decoder_attention_mask,
        min_length=1,
        max_length=4096,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        do_sample=False,
        num_beams=1,
    )
    
    sequence = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)[0]
    sequence = sequence.replace(prompt, "").replace("<pad>", "").replace("</s>", "").strip()
    return sequence

def answer_question(text: str, question: str) -> str:
    """Answer questions about the contract based on extracted text."""
    # Simple keyword-based answering (you might want to use a proper LLM here)
    question = question.lower()
    
    if "penalty" in question or "damages" in question:
        pattern = r"(?i)(penalty|damages|compensation).*?[\.\n]"
        matches = re.finditer(pattern, text)
        answers = [match.group(0).strip() for match in matches]
        return "\n".join(answers) if answers else "No specific penalty clause found."
    
    if "termination" in question:
        pattern = r"(?i)(termination|terminate).*?[\.\n]"
        matches = re.finditer(pattern, text)
        answers = [match.group(0).strip() for match in matches]
        return "\n".join(answers) if answers else "No specific termination clause found."
    
    return "I cannot find a specific answer to this question in the contract."

@app.post("/analyze/")
async def analyze_document(
    file: UploadFile = File(...),
    question: str = Form(None)
):
    contents = await file.read()
    
    # Handle PDF files
    if file.filename.lower().endswith('.pdf'):
        # Increase DPI for better OCR quality on PDFs
        images = convert_from_bytes(contents, dpi=300)
        full_text = ""
        for image in images:
            text = run_ocr(image)
            full_text += text + "\n\n--- Page End ---\n\n"
    else:
        # Handle single image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        full_text = run_ocr(image)
    
    # Extract clauses and analyze risks
    clauses = extract_clauses(full_text)
    risks = analyze_risks(clauses)
    
    # Answer specific question if provided
    answer = answer_question(full_text, question) if question else None
    
    return JSONResponse({
        "full_text": full_text,
        "clauses": clauses,
        "risks": risks,
        "answer": answer
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 