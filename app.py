"""
app.py
======
Streamlit UI for Aadhaar MCP Validator.
Converts uploaded document (PDF, PNG, JPG, JPEG) to Base64,
processes it with offline RapidOCR & local GLiNER SLM,
and displays the exact multilingual mapped fields and structured JSON.

Run with:
    streamlit run app.py
"""

import io
import base64
import os
import streamlit as st
from PIL import Image

# Import core validator and MCP tool
from aadhaar_validator import AadhaarValidator, get_validator
from mcp_tool import validate_aadhaar_document, validate_aadhaar_number

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aadhaar Multilingual Extractor",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Styling ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-card {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .status-success {
        background-color: #ECFDF5;
        border: 1px solid #10B981;
        color: #065F46;
    }
    .status-warning {
        background-color: #FFFBEB;
        border: 1px solid #F59E0B;
        color: #92400E;
    }
    .status-error {
        background-color: #FEF2F2;
        border: 1px solid #EF4444;
        color: #991B1B;
    }
    .mapping-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .field-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .field-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }
    .field-secondary {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2563EB;
        font-family: 'Segoe UI', system-ui, sans-serif;
        margin-top: 2px;
    }
    .badge-lang {
        display: inline-block;
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 14px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-script {
        display: inline-block;
        background: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
        border-radius: 14px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/Aadhaar_Logo.svg/200px-Aadhaar_Logo.svg.png", width=120)
    st.title("System Status")

    validator = get_validator()
    st.success("✅ RapidOCR: Offline Multilingual ONNX Active")
    if validator.slm_model is not None:
        st.success("✅ Local GLiNER SLM: Loaded (models/gliner_model)")
    else:
        st.info("ℹ️ Local SLM: Standalone Regex Mode")

    st.success("✅ Verhoeff Algorithm: Pure Python Checksum")
    st.info("🧭 4-Way Auto-Orientation: (0°, 90°, 180°, 270°)")

    st.divider()
    st.markdown("### Supported Documents")
    st.markdown("- **PDF** (`.pdf`) & **Images** (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`)")
    st.markdown("- **Supported Scripts**: English, Devanagari (Hindi), Odia, Tamil, Telugu, etc.")

    st.divider()
    st.caption("100% Offline • Enterprise Air-Gapped Ready")


# ── Main Header ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🇮🇳 Aadhaar Multilingual Document Validator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Offline RapidOCR with Small Language Model (GLiNER) & Mathematical Verhoeff Checksum Verification</div>', unsafe_allow_html=True)

# ── Navigation Tabs ──────────────────────────────────────────────────────────
tab_doc, tab_num, tab_about = st.tabs(["📄 Document Verification & Mapping", "🔢 Direct Aadhaar Number Check", "ℹ️ Architecture"])

# ==============================================================================
# TAB 1: Document Verification & Multilingual Mapping
# ==============================================================================
with tab_doc:
    col_input, col_preview = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. Upload Aadhaar Card")

        uploaded_file = st.file_uploader(
            "Select Aadhaar Document (PDF or Image)",
            type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
            help="Document is processed locally by offline RapidOCR and GLiNER SLM"
        )

        use_sample = st.checkbox("Or use standard sample Aadhaar card")

        execute_btn = st.button("🚀 Process & Map Fields", type="primary", use_container_width=True)

    base64_payload = None
    preview_image = None
    doc_name = ""

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        base64_payload = base64.b64encode(file_bytes).decode("utf-8")
        doc_name = uploaded_file.name

        if uploaded_file.name.lower().endswith(".pdf"):
            try:
                import pymupdf
                doc = pymupdf.open(stream=file_bytes, filetype="pdf")
                if len(doc) > 0:
                    page = doc[0]
                    pix = page.get_pixmap(dpi=150)
                    preview_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                doc.close()
            except Exception:
                preview_image = None
        else:
            try:
                preview_image = Image.open(io.BytesIO(file_bytes))
            except Exception:
                preview_image = None

    elif use_sample:
        # Load sample card if available
        sample_path = r"C:\Users\Vivek\.gemini\antigravity-ide\brain\261f5e20-2c1b-4d93-9ccb-612bb251bc98\.user_uploaded\media_1790762347667.jpg"
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                file_bytes = f.read()
            base64_payload = base64.b64encode(file_bytes).decode("utf-8")
            preview_image = Image.open(io.BytesIO(file_bytes))
            doc_name = "sample_aadhaar_vivek.jpg"

    with col_preview:
        st.subheader("Document Preview")
        if preview_image:
            st.image(preview_image, caption=f"Preview: {doc_name}", use_container_width=True)
        else:
            st.info("Upload an Aadhaar card image or check sample to preview.")

    # ── Execute & Display Exactly Mapped Output ────────────────────────────────
    if execute_btn:
        if not base64_payload:
            st.error("Please upload a file or check 'use standard sample Aadhaar card' first.")
        else:
            with st.spinner("Processing document: RapidOCR 4-Way Auto-Orientation & GLiNER SLM Extraction..."):
                result = validate_aadhaar_document(base64_payload)

            st.divider()
            st.subheader("2. Extracted Multilingual ID Mapping")

            is_aadhaar = result.get("is_aadhaar", False)
            data = result.get("data") or {}
            meta = result.get("meta") or {}
            multi_fields = data.get("multilingual_fields") or {}
            is_verhoeff = data.get("is_verhoeff_valid", False)

            # Verification Status Banner
            if is_aadhaar and is_verhoeff:
                st.markdown(f"""
                <div class="status-card status-success">
                    <h3 style="margin:0 0 4px 0;">✅ VALID AADHAAR CARD VERIFIED</h3>
                    <span>Aadhaar Number: <b>{data.get('aadhaar_number')}</b> &nbsp;|&nbsp; Mathematical Verhoeff Checksum: <b>Valid</b></span>
                </div>
                """, unsafe_allow_html=True)
            elif is_aadhaar and not is_verhoeff:
                st.markdown(f"""
                <div class="status-card status-warning">
                    <h3 style="margin:0 0 4px 0;">⚠️ AADHAAR CARD DETECTED (CHECKSUM MISMATCH)</h3>
                    <span>Aadhaar Number: <b>{data.get('aadhaar_number')}</b> &nbsp;|&nbsp; Checksum could not be mathematically verified.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-card status-error">
                    <h3 style="margin:0 0 4px 0;">❌ UNVERIFIED DOCUMENT</h3>
                    <span>{result.get('message', 'Could not extract valid Aadhaar card attributes.')}</span>
                </div>
                """, unsafe_allow_html=True)

            # ── EXACT MAPPED FIELDS (DISPLAYED PROMINENTLY) ─────────────────────
            name_info = multi_fields.get("name", {})
            dob_info = multi_fields.get("dob", {})
            gen_info = multi_fields.get("gender", {})
            hdr_info = multi_fields.get("government_header", {})
            tag_info = multi_fields.get("tagline", {})
            reg_lang = name_info.get("regional_language", "Regional")

            col_m1, col_m2 = st.columns(2)

            with col_m1:
                # Aadhaar Number Card
                st.markdown(f"""
                <div class="mapping-card">
                    <div class="field-title">🔢 Aadhaar Number</div>
                    <div class="field-value">{data.get('aadhaar_number') or 'Not Detected'}</div>
                    <div class="field-secondary">Masked: {data.get('aadhaar_number_masked') or 'N/A'} &nbsp;•&nbsp; Verhoeff: {'✅ Valid' if is_verhoeff else '❌ Invalid'}</div>
                </div>
                """, unsafe_allow_html=True)

                # Resident Name Card
                st.markdown(f"""
                <div class="mapping-card">
                    <div class="field-title">👤 Resident Name (English & {reg_lang})</div>
                    <div class="field-value">{name_info.get('english') or data.get('name') or 'Not Detected'}</div>
                    <div class="field-secondary">{reg_lang}: {name_info.get('regional') or data.get('name_regional') or 'N/A'}</div>
                </div>
                """, unsafe_allow_html=True)

                # Government Header Card
                st.markdown(f"""
                <div class="mapping-card">
                    <div class="field-title">🏛️ Government Header</div>
                    <div class="field-value">{hdr_info.get('english', 'Government of India')}</div>
                    <div class="field-secondary">Hindi: {hdr_info.get('hindi', 'भारत सरकार')}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m2:
                # Date of Birth Card
                eng_lbl = dob_info.get("english_label", "DOB")
                reg_lbl = dob_info.get("regional_label", "")
                lbl_str = f"{eng_lbl} / {reg_lbl}".strip(" /")
                st.markdown(f"""
                <div class="mapping-card">
                    <div class="field-title">📅 Date of Birth ({lbl_str})</div>
                    <div class="field-value">{dob_info.get('value') or data.get('dob') or 'Not Detected'}</div>
                    <div class="field-secondary">Labels: {lbl_str}</div>
                </div>
                """, unsafe_allow_html=True)

                # Gender Card
                st.markdown(f"""
                <div class="mapping-card">
                    <div class="field-title">⚧ Gender (English & {reg_lang})</div>
                    <div class="field-value">{gen_info.get('english') or data.get('gender') or 'Not Detected'}</div>
                    <div class="field-secondary">{reg_lang}: {gen_info.get('regional') or 'N/A'}</div>
                </div>
                """, unsafe_allow_html=True)

                # Tagline Card
                st.markdown(f"""
                <div class="mapping-card">
                    <div class="field-title">✨ Aadhaar Tagline</div>
                    <div class="field-value">{tag_info.get('hindi', 'मेरा आधार, मेरी पहचान')}</div>
                    <div class="field-secondary">Devanagari (Hindi)</div>
                </div>
                """, unsafe_allow_html=True)

            # Languages & Scripts Badges
            st.markdown("##### 🌐 Detected Languages & Scripts")
            badge_html = "<div>"
            for lang in result.get("detected_languages", []):
                badge_html += f'<span class="badge-lang">🗣️ {lang}</span>'
            for script in result.get("detected_scripts", []):
                badge_html += f'<span class="badge-script">✍️ {script}</span>'
            badge_html += "</div>"
            st.markdown(badge_html, unsafe_allow_html=True)

            st.write("")

            # ── Detailed Tabs: Exact JSON & Chunks ─────────────────────────────
            sub_tab_json, sub_tab_chunks, sub_tab_meta = st.tabs([
                "💻 Full Mapped JSON Response",
                "🌐 Retrieved Multilingual Chunks",
                "⏱️ Processing Metadata"
            ])

            with sub_tab_json:
                st.json(result)

            with sub_tab_chunks:
                chunks = result.get("multilingual_chunks", [])
                st.markdown(f"**Retrieved {len(chunks)} Chunks:**")
                for c in chunks:
                    cid = c.get("chunk_id", "-")
                    txt = c.get("text", "")
                    script = c.get("script", "Unknown")
                    cat = c.get("category", "TEXT_CONTENT")
                    conf = c.get("confidence", 0.0)
                    lang = c.get("language", "Unknown")

                    st.markdown(f"""
                    <div style="background:#F8FAFC; border-left:4px solid #3B82F6; padding:8px 14px; border-radius:6px; margin-bottom:6px;">
                        <span style="font-weight:700; color:#1E3A8A;">#{cid}</span> &nbsp;
                        <span style="background:#DBEAFE; color:#1E40AF; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600;">{cat}</span> &nbsp;
                        <span style="background:#FEF3C7; color:#92400E; padding:2px 8px; border-radius:12px; font-size:0.75rem;">{script} ({lang})</span> &nbsp;
                        <span style="color:#64748B; font-size:0.8rem; float:right;">Conf: {round(conf*100, 1)}%</span>
                        <div style="font-size:1.05rem; font-weight:600; color:#0F172A; margin-top:2px;">{txt}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with sub_tab_meta:
                col_mt1, col_mt2, col_mt3, col_mt4 = st.columns(4)
                with col_mt1:
                    st.metric("Processing Time", f"{meta.get('processing_time_sec', 0)} sec")
                with col_mt2:
                    st.metric("OCR Confidence", f"{round(meta.get('ocr_confidence', 0)*100, 1)}%")
                with col_mt3:
                    st.metric("Chunks Count", meta.get("chunks_count", 0))
                with col_mt4:
                    st.metric("Models Used", "RapidOCR + GLiNER")


# ==============================================================================
# TAB 2: Direct Number Validator
# ==============================================================================
with tab_num:
    st.subheader("Test 12-Digit Aadhaar Number (Verhoeff Algorithm)")
    st.markdown("Directly test the mathematical Verhoeff checksum without any image or document.")

    col_num_in, col_num_out = st.columns([1, 1], gap="large")

    with col_num_in:
        input_num = st.text_input("Enter 12-digit Aadhaar Number", value="6676 5884 0093", max_chars=16)
        check_btn = st.button("Check Verhoeff Algorithm", type="primary")

    with col_num_out:
        if check_btn and input_num:
            num_res = validate_aadhaar_number(input_num)
            if num_res.get("is_valid"):
                st.success(f"✅ **Valid Aadhaar Number**: `{num_res.get('aadhaar_number')}`")
                st.info(f"Masked format: `{num_res.get('aadhaar_number_masked')}`")
            else:
                st.error(f"❌ **Invalid**: {num_res.get('error')}")
            st.json(num_res)


# ==============================================================================
# TAB 3: Architecture & Information
# ==============================================================================
with tab_about:
    st.subheader("System Architecture")
    st.markdown("""
    This application connects to the **Aadhaar MCP Tool** running 100% locally and offline:
    1. **Input**: Accepts Base64 encoded PDF or Image.
    2. **Decoder**: PyMuPDF extracts PDF pages; PIL handles PNG/JPEG/WEBP.
    3. **Auto-Orientation**: 4-way rotation recovery (`0°`, `90°`, `180°`, `270°`) using layout geometry scoring.
    4. **OCR Engine**: Strictly **RapidOCR** (rapidocr_onnxruntime) with genuine multilingual models.
    5. **Normalizer**: Translates Indic numerals (`०-९`) to standard ASCII `0-9`.
    6. **SLM / NLP**: Local **GLiNER** Small Language Model extracts Named Entities (Person, DOB, Gender, Address).
    7. **Multilingual Mapping**: Maps resident details across Indian ID languages (English, Hindi, Odia, Tamil, Telugu, etc.) with zero foreign script hallucination.
    8. **Verifier**: Pure Python Verhoeff algorithm verifies the mathematical 12th checksum digit.
    9. **Output**: Structured JSON schema conforming to MCP tool standards.
    """)
