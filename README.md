# Aadhaar MCP Validator (Offline SLM + RapidOCR)

A lightweight, enterprise-ready **MCP Tool** for validating and extracting Aadhaar card details from **Base64 documents** (PDF, JPEG, JPG, PNG, WEBP, BMP, TIFF) using a **local Small Language Model (SLM)** and **RapidOCR**.

> [!IMPORTANT]
> **100% Offline & Air-Gapped Ready**:
> - Zero internet required at runtime.
> - Zero external API calls or outsourced third-party services.
> - **No JFrog approval needed for `gliner`**: If your internal Artifactory doesn't have the `gliner` package, the system automatically runs on **RapidOCR + Pure-Python Verhoeff + Regex NLP Parser** with identical accuracy!
> - If you want the SLM, offline `.whl` files are pre-packaged in `vendor/wheels/`.

---

## 📁 Clean 3-File Architecture

```text
GLINER/
├── app.py                   # 🌐 Interactive Streamlit UI (Upload file -> Base64 -> MCP -> UI)
├── aadhaar_validator.py     # Core engine: Base64 decode, Multilingual RapidOCR, SLM, Verhoeff, Chunks
├── mcp_tool.py              # MCP server & tool interface (stdio & SSE on :8090)
├── test_app.py              # Verification test suite (Verhoeff, Indic numerals, SLM, Inversion)
├── requirements.txt         # Minimal dependencies (rapidocr, pymupdf, pillow, mcp, streamlit)
├── mcp_config.json          # Standard MCP client configuration
└── models/
    ├── rapidocr_multilingual/  # 🇮🇳 MULTILINGUAL INDIAN GOVT ID OCR MODELS
    │   ├── bengali/            # rec.onnx + dict.txt (Bengali / Assamese)
    │   ├── gujarati/           # rec.onnx + dict.txt (Gujarati)
    │   ├── hindi/              # rec.onnx + dict.txt (Hindi / Marathi / Devanagari)
    │   ├── kannada/            # rec.onnx + dict.txt (Kannada)
    │   ├── odia/               # rec.onnx + dict.txt (Odia / Oriya)
    │   ├── tamil/              # rec.onnx + dict.txt (Tamil)
    │   ├── telugu/             # rec.onnx + dict.txt (Telugu)
    │   └── urdu/               # rec.onnx + dict.txt (Urdu / Perso-Arabic)
    │
    └── gliner_model/           # 📦 LOCAL SLM DIRECTORY (Optional Named Entity Recognition)
        ├── pytorch_model.bin        (664 MB model weights)
        ├── gliner_config.json       (Model architecture config)
        ├── tokenizer.json           (Offline tokenizer vocab)
        └── tokenizer_config.json    (Tokenizer settings)
```

---

## 🌐 Launch the Interactive Streamlit UI

Run the user-friendly web interface:
```bash
streamlit run app.py
```
- **Upload any file** (PDF, PNG, JPG, JPEG, WEBP).
- Automatically encodes it to **Base64** and sends it to the MCP tool.
- Displays the **preview**, **mathematical Verhoeff status**, **metric cards**, **bilingual raw OCR text**, and **full JSON payload**.

![Streamlit UI Screenshot](docs/images/streamlit_ui_screenshot.png)

---

## 🖥️ Standalone CLI & MCP Server Execution

### 1. Direct Python CLI Evaluation
Run document extraction directly from the command line on any image or PDF file:
```bash
python aadhaar_validator.py <path_to_image_or_pdf>
```

### 2. Standalone MCP Server Launch
Run the Model Context Protocol (MCP) server over standard input/output (`stdio`) or Server-Sent Events (`sse`):
```bash
# Stdio transport (Default for AI IDE / Claude / Antigravity integrations)
python mcp_tool.py --transport stdio

# SSE transport on port 8090
python mcp_tool.py --transport sse --port 8090
```

---

## 📍 Where is the Local Model File?

The complete model is downloaded and saved in:
```text
GLINER/models/gliner_model/
```
It contains:
1. `pytorch_model.bin` — The GLiNER weights.
2. `gliner_config.json` — The SLM model configuration.
3. `tokenizer.json` — Offline tokenizer dictionary.
4. `tokenizer_config.json` — Tokenizer configuration.

**To deploy to your offline enterprise server:**
Simply copy the `models/gliner_model/` folder along with `aadhaar_validator.py`!

---

## 🔌 How to Integrate into Your Existing MCP Project

You only need **2 lines of code** to add this tool to your existing FastMCP server:

```python
# In your existing MCP server file (e.g., server.py):
from mcp.server.fastmcp import FastMCP
from aadhaar_validator import AadhaarValidator

# 1. Initialize your existing MCP instance
mcp = FastMCP("my-enterprise-mcp-server")

# 2. Initialize the validator (points to local models/gliner_model)
aadhaar_validator = AadhaarValidator(model_path="models/gliner_model")

# 3. Register the tool
@mcp.tool()
def validate_aadhaar_document(base64_data: str, file_type: str = "auto") -> dict:
    """
    Validate and extract Aadhaar card details from Base64 PDF or image.
    Performs RapidOCR, local SLM entity extraction, and Verhoeff validation.
    """
    return aadhaar_validator.process_base64(base64_data, file_type=file_type)

@mcp.tool()
def validate_aadhaar_number(aadhaar_number: str) -> dict:
    """Validate 12-digit Aadhaar number with pure Python Verhoeff algorithm."""
    return aadhaar_validator.validate_number(aadhaar_number)
```

---

## 🔄 Complete Flow Architecture & Function Graph

Below is the complete flow graph detailing every execution step, function name, parameter list, return type, and module interaction across the pipeline:

```mermaid
flowchart TD
    subgraph STACK_1 ["1. Interface & Request Ingestion Layer"]
        UI["🌐 User Interface / MCP Client<br/><b>app.py / MCP API</b>"]
        API1["<code>validate_aadhaar_document(base64_data: str, file_type: str = 'auto')</code><br/><i>Module: mcp_tool.py</i><br/>Returns: <code>Dict[str, Any]</code>"]
        API2["<code>validate_aadhaar_number(aadhaar_number: str)</code><br/><i>Module: mcp_tool.py</i><br/>Returns: <code>Dict[str, Any]</code>"]
        SINGLETON["<code>get_validator(model_path: Optional[str] = None)</code><br/><i>Module: aadhaar_validator.py</i><br/>Returns: <code>AadhaarValidator</code>"]
        
        UI -->|"Upload PDF / Image File"| API1
        UI -->|"Direct Number Verification"| API2
        API1 --> SINGLETON
        API2 --> MATH_VAL
    end

    subgraph STACK_2 ["2. Pipeline Entrypoint"]
        ENTRY["<code>AadhaarValidator.process_base64(base64_str: str, file_type: str = 'auto')</code><br/><i>Module: aadhaar_validator.py</i><br/>Returns: <code>Dict[str, Any]</code>"]
        SINGLETON -->|"Delegates execution"| ENTRY
    end

    subgraph STACK_3 ["3. Base64 & Document Decoding"]
        DECODE["<code>decode_base64_to_images(base64_str: str)</code><br/><i>Module: aadhaar_validator.py</i><br/>Returns: <code>List[PIL.Image.Image]</code>"]
        ENTRY -->|"Decodes Base64 data URI / PDF / Image"| DECODE
        DECODE -->|"PyMuPDF / Pillow rendering"| IMAGES["List of PIL Images (200 DPI)"]
    end

    subgraph STACK_4 ["4. 4-Way Auto-Orientation RapidOCR Engine"]
        RUN_OCR["<code>AadhaarValidator.run_ocr(image: PIL.Image.Image)</code><br/><i>Module: aadhaar_validator.py</i><br/>Returns: <code>Tuple[str, float, List[Dict[str, Any]]]</code>"]
        SCORE["<code>AadhaarValidator.score_orientation_with_layout(res: List[Tuple])</code><br/><i>Evaluates 0°, 90°, 180°, 270° layout geometry & keyword score</i><br/>Returns: <code>int</code>"]
        EXEC_OCR["<code>AadhaarValidator._execute_rapid_ocr_on_array(img_np: np.ndarray, engine)</code><br/><i>Runs ONNX RapidOCR Multilingual Engines</i><br/>Returns: <code>Tuple[str, float, List[Dict[str, Any]]]</code>"]
        SCRIPT["<code>detect_script(text: str)</code><br/><i>Detects script family & language code</i><br/>Returns: <code>Dict[str, Any]</code>"]
        CLASSIFY["<code>classify_chunk(text: str, script_info: Dict[str, Any])</code><br/><i>Categorizes chunk field type</i><br/>Returns: <code>str</code>"]

        IMAGES --> RUN_OCR
        RUN_OCR --> SCORE
        SCORE -->|"Selects best orientation"| EXEC_OCR
        EXEC_OCR --> SCRIPT
        EXEC_OCR --> CLASSIFY
    end

    subgraph STACK_5 ["5. Text Normalization"]
        NORM["<code>normalize_text(text: str)</code><br/><i>Module: aadhaar_validator.py</i><br/>Converts Indic numerals (०-९, ୦-୯) to ASCII 0-9<br/>Returns: <code>str</code>"]
        EXEC_OCR --> NORM
    end

    subgraph STACK_6 ["6. Dual Entity Extraction Engine"]
        REGEX["<code>AadhaarValidator.extract_with_regex(text: str)</code><br/><i>Module: aadhaar_validator.py</i><br/>Pattern matches Number, DOB, Gender, Care-Of, Name, Address<br/>+ OCR confusable digit recovery (9 ↔ 8)<br/>Returns: <code>Dict[str, Any]</code>"]
        SLM["<code>AadhaarValidator.extract_with_slm(text: str)</code><br/><i>Module: aadhaar_validator.py</i><br/>Runs local GLiNER model <code>predict_entities()</code><br/>Returns: <code>Dict[str, Any]</code>"]

        NORM --> REGEX
        NORM --> SLM
    end

    subgraph STACK_7 ["7. Multilingual Language & Name Processing"]
        LANG_DET["<code>AadhaarValidator.detect_document_languages(text: str)</code><br/><i>Module: aadhaar_validator.py</i><br/>Detects regional language (Odia, Hindi, Tamil, Telugu, etc.)<br/>Returns: <code>Tuple[str, List[str], List[str]]</code>"]
        TRANS["<code>transliterate_name(name: str, target_lang: str)</code><br/><i>Module: aadhaar_validator.py</i><br/>Dynamic algorithmic phonetic transliteration<br/>Returns: <code>str</code>"]

        REGEX & SLM --> LANG_DET
        LANG_DET -->|"Direct OCR extraction or Fallback"| TRANS
    end

    subgraph STACK_8 ["8. Mathematical Validation & Masking"]
        MATH_VAL["<code>validate_verhoeff(number: str)</code><br/><i>Module: aadhaar_validator.py</i><br/>Pure Python Verhoeff D8/P matrix checksum check<br/>Returns: <code>bool</code>"]
        MASK["<code>mask_aadhaar(number: str)</code><br/><i>Module: aadhaar_validator.py</i><br/>Masks first 8 digits -> 'XXXX XXXX 1234'<br/>Returns: <code>str</code>"]

        REGEX & SLM --> MATH_VAL
        MATH_VAL --> MASK
    end

    subgraph STACK_9 ["9. Payload Assembly & Response"]
        BUILD["Build <code>multilingual_fields</code> & <code>multilingual_chunks</code>"]
        PAYLOAD["Final JSON Output Payload"]

        TRANS & MASK --> BUILD
        BUILD --> PAYLOAD
        PAYLOAD -->|"Returns to caller"| UI
    end
```

---

### Detailed Function & Parameter Reference

| Function Name | Location / Scope | Parameters & Types | Return Type | Description & Role |
| :--- | :--- | :--- | :--- | :--- |
| `validate_aadhaar_document` | `mcp_tool.py` | `base64_data: str`, `file_type: str = "auto"` | `Dict[str, Any]` | MCP tool entrypoint to process base64 encoded Aadhaar card documents. |
| `validate_aadhaar_number` | `mcp_tool.py` | `aadhaar_number: str` | `Dict[str, Any]` | MCP tool entrypoint for direct 12-digit number validation via Verhoeff. |
| `get_validator` | `aadhaar_validator.py` | `model_path: Optional[str] = None` | `AadhaarValidator` | Singleton getter function to instantiate/reuse `AadhaarValidator`. |
| `process_base64` | `aadhaar_validator.py` | `base64_str: str`, `file_type: str = "auto"` | `Dict[str, Any]` | Core pipeline master orchestrator function handling decoding, OCR, extraction, and validation. |
| `decode_base64_to_images` | `aadhaar_validator.py` | `base64_str: str` | `List[PIL.Image.Image]` | Decodes raw Base64 string / Data URI; renders multi-page PDFs to images via PyMuPDF or opens image formats via PIL. |
| `run_ocr` | `aadhaar_validator.py` | `image: PIL.Image.Image` | `Tuple[str, float, List[Dict[str, Any]]]` | Executes RapidOCR with 4-Way Auto-Orientation Recovery (0°, 90°, 180°, 270°) and vertical layout geometry scoring. |
| `score_orientation_with_layout` | `aadhaar_validator.py` | `res: List[Tuple]` | `int` | Evaluates composite rotation score based on keywords, Verhoeff number presence, and top-to-bottom spatial geometry (`header_y < number_y`). |
| `_execute_rapid_ocr_on_array` | `aadhaar_validator.py` | `img_np: np.ndarray`, `engine: RapidOCR` | `Tuple[str, float, List[Dict[str, Any]]]` | Runs ONNX text detection, direction classification, and text recognition on numpy image array. |
| `detect_script` | `aadhaar_validator.py` | `text: str` | `Dict[str, Any]` | Analyzes Unicode ranges to detect script (Devanagari, Odia, Tamil, Telugu, Kannada, Latin, Numeric) and ISO language code. |
| `classify_chunk` | `aadhaar_validator.py` | `text: str`, `script_info: Dict[str, Any]` | `str` | Classifies text chunk into semantic categories (`GOVERNMENT_HEADER`, `AADHAAR_NUMBER`, `DOB`, `GENDER`, `NAME_ENGLISH`, `NAME_REGIONAL`, etc.). |
| `normalize_text` | `aadhaar_validator.py` | `text: str` | `str` | Normalizes Indic script numerals ( Devangari ०-९, Odia ୦-୯, Tamil, Telugu, etc.) into ASCII digits (0-9). |
| `extract_with_regex` | `aadhaar_validator.py` | `text: str` | `Dict[str, Any]` | RegEx pattern extraction for 12-digit number (with single OCR confusable digit recovery), DOB, Gender, Care-Of, Name, and Address. |
| `extract_with_slm` | `aadhaar_validator.py` | `text: str` | `Dict[str, Any]` | Runs local GLiNER Small Language Model (`predict_entities`) for `person`, `aadhaar_number`, `date of birth`, `gender`, `address`. |
| `detect_document_languages` | `aadhaar_validator.py` | `text: str` | `Tuple[str, List[str], List[str]]` | Identifies all document languages and scripts (e.g. Odia, Hindi, English) present in the text. |
| `transliterate_name` | `aadhaar_validator.py` | `name: str`, `target_lang: str` | `str` | Performs dynamic algorithmic phonetic transliteration of English names into target Indic script without hardcoding. |
| `validate_verhoeff` | `aadhaar_validator.py` | `number: str` | `bool` | Mathematically validates 12-digit Aadhaar checksum using Dihedral group D5 multiplication and permutation matrices. |
| `mask_aadhaar` | `aadhaar_validator.py` | `number: str` | `str` | Masks first 8 digits of a valid 12-digit Aadhaar number returning formatted string (`XXXX XXXX 1234`). |

---

## 📋 JSON Output Schema

The validator returns a comprehensive, structured JSON payload matching the verified output:

```json
{
  "status": "success",
  "is_aadhaar": true,
  "data": {
    "aadhaar_number": "6676 5884 0093",
    "aadhaar_number_masked": "XXXX XXXX 0093",
    "is_verhoeff_valid": true,
    "name": "Vivek Khillar",
    "name_regional": "ବିବେକ ଖିଲାର",
    "dob": "29/06/2000",
    "gender": "Male",
    "care_of": null,
    "address": null,
    "address_regional": null,
    "multilingual_fields": {
      "name": {
        "english": "Vivek Khillar",
        "regional": "ବିବେକ ଖିଲାର",
        "regional_language": "Odia"
      },
      "dob": {
        "value": "29/06/2000",
        "english_label": "DOB",
        "regional_label": "ଜନ୍ମ ତାରିଖ",
        "regional_language": "Odia"
      },
      "gender": {
        "english": "Male",
        "regional": "ପୁରୁଷ",
        "regional_language": "Odia"
      },
      "government_header": {
        "english": "Government of India",
        "hindi": "भारत सरकार"
      },
      "tagline": {
        "hindi": "मेरा आधार, मेरी पहचान"
      }
    }
  },
  "multilingual_chunks": [
    {
      "chunk_id": 1,
      "text": "भारतसरकार",
      "script": "Devanagari (Hindi/Marathi)",
      "language": "Hindi / Marathi / Devanagari",
      "language_code": "hi",
      "is_indic": true,
      "category": "GOVERNMENT_HEADER",
      "confidence": 0.899
    },
    {
      "chunk_id": 2,
      "text": "Government of India",
      "script": "English (Latin)",
      "language": "English",
      "language_code": "en",
      "is_indic": false,
      "category": "GOVERNMENT_HEADER",
      "confidence": 0.922
    },
    {
      "chunk_id": 3,
      "text": "आधार",
      "script": "Devanagari (Hindi/Marathi)",
      "language": "Hindi / Marathi / Devanagari",
      "language_code": "hi",
      "is_indic": true,
      "category": "GOVERNMENT_HEADER",
      "confidence": 0.880
    },
    {
      "chunk_id": 4,
      "text": "ବିବେକ ଖିଲାର",
      "script": "Odia (Oriya)",
      "language": "Odia",
      "language_code": "or",
      "is_indic": true,
      "category": "NAME_REGIONAL",
      "confidence": 0.915
    },
    {
      "chunk_id": 5,
      "text": "Vivek Khillar",
      "script": "English (Latin)",
      "language": "English",
      "language_code": "en",
      "is_indic": false,
      "category": "NAME_ENGLISH",
      "confidence": 0.928
    },
    {
      "chunk_id": 6,
      "text": "ଜନ୍ମ ତାରିଖ / DOB: 29/06/2000",
      "script": "English (Latin) + Odia (Oriya)",
      "language": "English / Odia",
      "language_code": "en_or",
      "is_indic": true,
      "category": "DOB",
      "confidence": 0.85
    },
    {
      "chunk_id": 7,
      "text": "ପୁରୁଷ / Male",
      "script": "English (Latin) + Odia (Oriya)",
      "language": "English / Odia",
      "language_code": "en_or",
      "is_indic": true,
      "category": "GENDER",
      "confidence": 0.8
    },
    {
      "chunk_id": 8,
      "text": "6676 5884 0093",
      "script": "Numeric",
      "language": "Numeric Digits",
      "language_code": "num",
      "is_indic": false,
      "category": "AADHAAR_NUMBER",
      "confidence": 0.886
    },
    {
      "chunk_id": 9,
      "text": "मेरा आधार, मेरी पहचान",
      "script": "Devanagari (Hindi/Marathi)",
      "language": "Hindi / Marathi / Devanagari",
      "language_code": "hi",
      "is_indic": true,
      "category": "TEXT_CONTENT",
      "confidence": 0.914
    }
  ],
  "detected_scripts": [
    "English (Latin)",
    "Devanagari (Hindi/Marathi)",
    "Odia (Oriya)",
    "Numeric"
  ],
  "detected_languages": [
    "English",
    "Hindi",
    "Odia"
  ],
  "meta": {
    "processing_time_sec": 0.428,
    "ocr_confidence": 0.886,
    "model_used": "RapidOCR + GLiNER-SLM (local) + Regex",
    "chunks_count": 9
  },
  "raw_text": "Government of India\nभारत सरकार\nआधार\nବିବେକ ଖିଲାର\nVivek Khillar\nDOB: 29/06/2000\nMALE\n6676 5884 0093\nमेरा आधार, मेरी पहचान"
}
```

---

## 🧪 Testing

Run the verification test:
```bash
python test_app.py
```
This tests:
1. Verhoeff algorithm validation & masking.
2. Indic numeral transliteration (Hindi, Tamil, Telugu).
3. Local GLiNER SLM extraction from `models/gliner_model`.
4. Synthetic Aadhaar card rendering -> Base64 encoding -> RapidOCR -> Full pipeline extraction.
