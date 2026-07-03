import os
import io
import re
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import PyPDF2
import docx
from PIL import Image
import pytesseract
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "EeupnxMTWnsZBCskMPRE9riA6L8srSJ9")
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-tiny"

# In-memory storage
user_data = {
    "medications": [],
    "appointments": [],
    "labs": [],
    "files": [],
    "chat_history": []
}

# ------------------- Helper Functions -------------------
def call_mistral(messages):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 800,
        "top_p": 1
    }
    try:
        resp = requests.post(MISTRAL_ENDPOINT, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Mistral API error: {e}")
        return f"⚠️ Error calling Mistral AI: {str(e)}"

def extract_text_from_pdf(file_bytes):
    try:
        with io.BytesIO(file_bytes) as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        raise Exception(f"PDF extraction failed: {e}")

def extract_text_from_docx(file_bytes):
    try:
        with io.BytesIO(file_bytes) as docx_file:
            doc = docx.Document(docx_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
    except Exception as e:
        raise Exception(f"DOCX extraction failed: {e}")

def extract_text_from_image(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes))
        try:
            text = pytesseract.image_to_string(image)
            return text
        except Exception:
            return "Image uploaded. OCR not available; please describe the image contents for analysis."
    except Exception as e:
        raise Exception(f"Image processing failed: {e}")

def process_uploaded_file(file):
    filename = file.filename
    file_bytes = file.read()
    file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
    file_size = len(file_bytes)

    extracted_text = ""
    if file_ext in ['pdf']:
        extracted_text = extract_text_from_pdf(file_bytes)
    elif file_ext in ['docx', 'doc']:
        extracted_text = extract_text_from_docx(file_bytes)
    elif file_ext in ['txt']:
        extracted_text = file_bytes.decode('utf-8', errors='ignore')
    elif file_ext in ['png', 'jpg', 'jpeg', 'gif']:
        extracted_text = extract_text_from_image(file_bytes)
    else:
        raise Exception(f"Unsupported file type: {file_ext}")

    return {
        "filename": filename,
        "extension": file_ext,
        "size": file_size,
        "text": extracted_text[:8000]  # keep full text for analysis
    }

def analyze_document_with_ai(text, filename):
    prompt = f"""You are a medical document analyzer. Analyze the following medical document and provide:

1. **Document Type:** What kind of medical document is this? (e.g., Lab Report, Prescription, Medical History, Discharge Summary, etc.)
2. **Key Findings:** List the most important medical findings.
3. **Possible Diagnosis:** Based on the findings, what condition(s) does the patient likely have? Be specific.
4. **Abnormal Values:** List any abnormal lab values or concerning signs.
5. **Summary:** Provide a plain-language summary for a patient.
6. **Recommendations:** What actions should the patient consider?

Document Name: {filename}

Document Content:
{text}

Provide a clear, structured analysis with clear labels."""
    messages = [
        {"role": "system", "content": "You are a medical document analyzer. Provide accurate, clear, and safe analysis."},
        {"role": "user", "content": prompt}
    ]
    return call_mistral(messages)

def detect_appointment_from_text(text):
    lower = text.lower()
    if any(kw in lower for kw in ['appointment', 'schedule', 'visit', 'consultation', 'with dr', 'with doctor']):
        date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})', text)
        if not date_match:
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
        appt_date = date_match.group(1) if date_match else datetime.now().strftime("%Y-%m-%d")
        provider_match = re.search(r'with\s+(Dr\.?\s*\w+)', text, re.IGNORECASE)
        provider = provider_match.group(1) if provider_match else "Healthcare Provider"
        return {"date": appt_date, "provider": provider}
    return None

# ------------------- Routes -------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    system_prompt = data.get('system_prompt', '')
    history = data.get('history', [])
    # Check if the user asks about a file
    file_context = ""
    if user_data['files'] and any(kw in user_message.lower() for kw in ['report', 'document', 'file', 'analysis', 'this', 'it']):
        # Use the most recent file's text as context
        latest_file = user_data['files'][-1]
        file_context = f"\n\n**Recent uploaded document:** {latest_file['name']}\n\nContent:\n{latest_file['text'][:4000]}\n\nBased on this document, please answer the user's question."

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # Add file context as a system message if available
    if file_context:
        messages.append({"role": "system", "content": file_context})
    for msg in history:
        if msg.get('role') in ['user', 'assistant']:
            messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    response = call_mistral(messages)
    return jsonify({"response": response})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        processed = process_uploaded_file(file)
        analysis = analyze_document_with_ai(processed['text'], processed['filename'])

        is_lab = any(kw in analysis.lower() for kw in ['lab', 'blood', 'test', 'result'])
        is_appt = any(kw in analysis.lower() for kw in ['appointment', 'schedule', 'visit'])

        file_entry = {
            "id": int(datetime.now().timestamp()),
            "name": processed['filename'],
            "type": processed['extension'],
            "size": processed['size'],
            "text": processed['text'],  # store full text
            "analysis": analysis,
            "uploadedAt": datetime.now().isoformat()
        }
        user_data['files'].append(file_entry)

        if is_lab:
            lab_entry = {
                "title": f"📄 {processed['filename']}",
                "raw": processed['text'][:1000],
                "summary": analysis,
                "date": datetime.now().strftime("%b %d, %Y"),
                "fromFile": True,
                "fileId": file_entry['id']
            }
            user_data['labs'].append(lab_entry)

        if is_appt:
            appt_info = detect_appointment_from_text(processed['text'])
            if not appt_info:
                appt_info = {"date": datetime.now().strftime("%Y-%m-%d"), "provider": "Healthcare Provider"}
            appt_entry = {
                "title": f"📄 {processed['filename']}",
                "provider": appt_info['provider'],
                "date": appt_info['date'],
                "time": "09:00",
                "fromFile": True
            }
            user_data['appointments'].append(appt_entry)

        return jsonify({
            "success": True,
            "file": file_entry,
            "analysis": analysis,
            "isLab": is_lab,
            "isAppointment": is_appt
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify({
        "medications": user_data['medications'],
        "appointments": user_data['appointments'],
        "labs": user_data['labs'],
        "files": user_data['files']
    })

@app.route('/api/medications', methods=['POST'])
def add_medication():
    data = request.get_json()
    if not data or not data.get('name'):
        return jsonify({"error": "Missing medication name"}), 400
    med = {
        "name": data['name'],
        "dosage": data.get('dosage', 'As prescribed'),
        "frequency": data.get('frequency', 'Daily'),
        "time": data.get('time', '08:00')
    }
    user_data['medications'].append(med)
    return jsonify({"success": True, "medication": med})

@app.route('/api/medications/<int:index>', methods=['DELETE'])
def delete_medication(index):
    if 0 <= index < len(user_data['medications']):
        removed = user_data['medications'].pop(index)
        return jsonify({"success": True, "removed": removed})
    return jsonify({"error": "Index out of range"}), 404

@app.route('/api/appointments', methods=['POST'])
def add_appointment():
    data = request.get_json()
    if not data or not data.get('title'):
        return jsonify({"error": "Missing appointment title"}), 400
    appt = {
        "title": data['title'],
        "provider": data.get('provider', 'Healthcare Provider'),
        "date": data.get('date', datetime.now().strftime("%Y-%m-%d")),
        "time": data.get('time', '09:00')
    }
    user_data['appointments'].append(appt)
    return jsonify({"success": True, "appointment": appt})

@app.route('/api/appointments/<int:index>', methods=['DELETE'])
def delete_appointment(index):
    if 0 <= index < len(user_data['appointments']):
        removed = user_data['appointments'].pop(index)
        return jsonify({"success": True, "removed": removed})
    return jsonify({"error": "Index out of range"}), 404

@app.route('/api/labs', methods=['POST'])
def add_lab():
    data = request.get_json()
    if not data or not data.get('title'):
        return jsonify({"error": "Missing lab title"}), 400
    lab = {
        "title": data['title'],
        "raw": data.get('raw', ''),
        "summary": data.get('summary', ''),
        "date": data.get('date', datetime.now().strftime("%b %d, %Y"))
    }
    user_data['labs'].append(lab)
    return jsonify({"success": True, "lab": lab})

@app.route('/api/labs/<int:index>', methods=['DELETE'])
def delete_lab(index):
    if 0 <= index < len(user_data['labs']):
        removed = user_data['labs'].pop(index)
        return jsonify({"success": True, "removed": removed})
    return jsonify({"error": "Index out of range"}), 404

@app.route('/api/files/<int:index>', methods=['DELETE'])
def delete_file(index):
    if 0 <= index < len(user_data['files']):
        removed = user_data['files'].pop(index)
        return jsonify({"success": True, "removed": removed})
    return jsonify({"error": "Index out of range"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)