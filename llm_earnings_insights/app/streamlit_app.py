import sys
from pathlib import Path

# make sure project root (one folder up) is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("ROOT:", ROOT)
print("TOP OF sys.path:", sys.path[:3])


import streamlit as st
from core.parse_pdf import extract_pdf_text
from core.chunk_index import ChunkIndex
from core.extract_llm import extract_earnings_info

st.title("LLM Earnings Insights")
st.caption("A personal project by Eemil Ylä-Nikkilä")

uploaded = st.file_uploader("Upload earnings report (PDF)", type=["pdf"])

if uploaded:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded.read())

    st.write("Extracting text...")
    pages = extract_pdf_text("temp.pdf")

    st.write(f"Parsed {len(pages)} pages")

    # Build index
    idx = ChunkIndex()
    idx.build(pages)

    st.write("Querying LLM for KPIs...")
    joined_text = " ".join([txt for _, txt in pages[:3]])  # just first 3 pages for MVP
    extract = extract_earnings_info(joined_text)

    st.json(extract.dict() if hasattr(extract, "dict") else extract)
