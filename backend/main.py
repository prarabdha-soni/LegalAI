import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_bytes
import re
import pandas as pd

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

def clean_latex_output(text: str) -> str:
    """
    Cleans up OCR output if it's in a LaTeX format by extracting
    text from \\text{...} blocks.
    """
    if text.strip().startswith("$$\\begin{aligned}"):
        # Find all text within \text{...}
        matches = re.findall(r"\\text\s*\{(.*?)\}", text)
        if matches:
            # Join the found text and clean up artifacts
            cleaned_text = " ".join(matches)
            # Remove any remaining LaTeX-like artifacts
            cleaned_text = re.sub(r"\\(?!n)", "", cleaned_text) # removes single backslashes not followed by 'n'
            return cleaned_text.strip()
    return text

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

def parse_sanction_letter(text):
    data = {
        "Name": "",
        "Loan Amount": "",
        "Tenor": "",
        "Interest Rate": "",
        "Processing Fees": "",
        "Disbursement Amount": "",
        "Overdue Charges": "",
        "Sanction Letter Validity": "",
    }
    for line in text.splitlines():
        if "Name:" in line:
            data["Name"] = line.split("Name:")[-1].strip()
        if "Sanction Amount" in line:
            data["Loan Amount"] = line.split("Sanction Amount (Total)")[-1].strip() if "Sanction Amount (Total)" in line else ""
        if "Loan Tenor" in line:
            data["Tenor"] = line.split("Loan Tenor")[-1].strip()
        if "Rate of Interest" in line:
            data["Interest Rate"] = line.split("Rate of Interest")[-1].strip()
        if "Processing Fees" in line:
            data["Processing Fees"] = line.split("Processing Fees")[-1].strip()
        if "Disbursement amount" in line:
            data["Disbursement Amount"] = line.split("Disbursement amount")[-1].strip()
        if "Overdue Charges" in line:
            data["Overdue Charges"] = line.split("Overdue Charges")[-1].strip()
        if "Sanction Letter Validity" in line:
            data["Sanction Letter Validity"] = line.split("Sanction Letter Validity")[-1].strip()
    return data

@app.post("/analyze/")
async def analyze_document(
    file: UploadFile = File(...),
    question: str = Form(None)
):
    contents = await file.read()
    filename = file.filename.lower()
    
    # Handle PDF files
    if filename.endswith('.pdf'):
        # Increase DPI for better OCR quality on PDFs
        images = convert_from_bytes(contents, dpi=300)
        full_text = ""
        for image in images:
            text = run_ocr(image) # Use the simple, direct prompt
            full_text += text + "\n\n--- Page End ---\n\n"
    else:
        # Handle single image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        # Use a specific prompt for Aadhar cards
        if "aadhar" in filename:
            prompt = "Extract all text from this identity card."
        else:
            prompt = "Read text in the image."
        full_text = run_ocr(image, prompt=prompt)
    
    # Clean the OCR output in case of LaTeX format
    full_text = clean_latex_output(full_text)
    
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

@app.post("/extract_sanction_excel/")
async def extract_sanction_excel(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    text = run_ocr(image, prompt="Extract all text from this sanction letter.")
    text = clean_latex_output(text)
    data = parse_sanction_letter(text)
    df = pd.DataFrame([data])
    excel_path = "sanction_letter.xlsx"
    df.to_excel(excel_path, index=False)
    return FileResponse(excel_path, filename="sanction_letter.xlsx")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 