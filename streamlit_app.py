# streamlit_app.py 
# Zero-Carbon Procurement AI — MVP
# Universal upload + OCR + Dual Upload Modes + Filter + NLQ v3  (no operator selector)

import io
import re
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# -------- Optional deps (guarded) --------
try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None


# ---------------- Page setup ----------------
st.set_page_config(page_title="Zero-Carbon Procurement AI — MVP", layout="wide")
st.markdown(
    """
    <style>
      .centered-area{max-width:1100px;margin:0 auto}
      .file-card{border:1px solid rgba(0,0,0,.12);border-radius:14px;padding:14px;background:#fff;
                 box-shadow:0 1px 3px rgba(0,0,0,.03);text-align:center;height:120px;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='centered-area'><h1 style='text-align:center;margin-top:6px;'>ZERO-CARBON PROCUREMENT AI — MVP</h1></div>",
    unsafe_allow_html=True,
)

# ---------------- State ----------------
if "files" not in st.session_state:
    st.session_state.files: List[Dict[str, Any]] = []
if "messages" not in st.session_state:
    st.session_state.messages: List[Tuple[str, str]] = []
if "analysis" not in st.session_state:
    st.session_state.analysis: Dict[str, Any] = {"items": pd.DataFrame()}
if "co2_factors_df" not in st.session_state:
    st.session_state.co2_factors_df: Optional[pd.DataFrame] = None

def toast(msg: str):
    try:
        st.toast(msg)
    except Exception:
        st.info(msg)

# -------------- OCR config ---------------
# If not on PATH, uncomment & set explicitly:
# if pytesseract: pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OCR_LANG = "eng"  # e.g., "eng+ara" if bilingual files


# -------------- Parsers ------------------
def read_pdf_text(b: bytes) -> str:
    if not PdfReader:
        return "[PDF parser not installed]"
    try:
        with io.BytesIO(b) as f:
            reader = PdfReader(f)
            texts = []
            for p in reader.pages:
                try:
                    texts.append(p.extract_text() or "")
                except Exception:
                    pass
            return "\n".join(texts).strip()
    except Exception as e:
        return f"[PDF read error: {e}]"

def read_pdf_scanned_ocr(b: bytes, max_pages=20) -> str:
    if not convert_from_bytes:
        return "[pdf2image not installed]"
    if not (Image and pytesseract):
        return "[OCR dependencies not installed]"
    try:
        pages = convert_from_bytes(b, dpi=200)
        out = []
        for i, img in enumerate(pages):
            if i >= max_pages:
                out.append("[…truncated remaining pages for speed…]")
                break
            try:
                out.append(pytesseract.image_to_string(img, lang=OCR_LANG))
            except Exception as e:
                out.append(f"[OCR error: {e}]")
        return "\n".join(out).strip()
    except Exception as e:
        return f"[PDF OCR fallback error: {e}]"

def read_excel_all_sheets(b: bytes) -> pd.DataFrame:
    try:
        with io.BytesIO(b) as f:
            xl = pd.ExcelFile(f)
            # auto-ignore summary/pivot helper sheets if present
            sheets = [s for s in xl.sheet_names if not s.lower().startswith(("import_", "summary", "pivot"))]
            if not sheets:
                sheets = xl.sheet_names
            frames = []
            for sh in sheets:
                try:
                    df = xl.parse(sh)
                    df["__sheet__"] = sh
                    frames.append(df)
                except Exception:
                    pass
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception as e:
        st.warning(f"Excel load error: {e}")
        return pd.DataFrame()

def read_image_ocr(b: bytes) -> str:
    if not Image:
        return "[PIL not installed]"
    if not pytesseract:
        return "[pytesseract not installed]"
    try:
        img = Image.open(io.BytesIO(b))
        return pytesseract.image_to_string(img, lang=OCR_LANG)
    except Exception as e:
        return f"[OCR error: {e}]"

def decode_binary(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return "[binary file: cannot decode]"


# -------- Heuristic extraction ----------
ITEM_LINE = re.compile(r"(?i)(pump|compressor|valve|filter|separator|vessel|meter|heater|chiller|dryer|blower|generator|transformer|switchgear|motor|actuator|skid|package|bundle|tube|pipe|fitting|gasket|gauge|sensor|wellhead|bit)")
MANU_HINT = re.compile(r"(?i)(manufacturer|vendor|make|brand)\s*[:\-]\s*([A-Za-z0-9 \-/&._]+)")
PRICE_HINT = re.compile(r"(?i)(price|unit price|amount|cost)\s*[:\-]?\s*\$?\s*([0-9][0-9,.\s]*)")
MATERIAL_HINT = re.compile(r"(?i)(material|MOC|M\.O\.C\.)\s*[:\-]\s*([A-Za-z0-9 \-/&._]+)")

def df_to_lines(df: pd.DataFrame) -> List[str]:
    rows = []
    if df.empty:
        return rows
    for _, r in df.iterrows():
        vals = [str(v) for v in r.values if pd.notna(v) and str(v).strip()]
        if vals:
            rows.append(" | ".join(vals))
    return rows

def extract_from_text(lines: List[str]) -> List[Dict[str, Any]]:
    items = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if ITEM_LINE.search(s):
            manu = MANU_HINT.search(s).group(2).strip() if MANU_HINT.search(s) else ""
            mat = MATERIAL_HINT.search(s).group(2).strip() if MATERIAL_HINT.search(s) else ""
            price = PRICE_HINT.search(s).group(2).strip() if PRICE_HINT.search(s) else ""
            items.append({
                "category": "", "subcategory": "",
                "item_name": s[:200],
                "material": mat, "manufacturer_name": manu, "price": price
            })
        else:
            if items:
                last = items[-1]
                if not last.get("manufacturer_name"):
                    mh = MANU_HINT.search(s)
                    if mh:
                        last["manufacturer_name"] = mh.group(2).strip()
                if not last.get("material"):
                    mth = MATERIAL_HINT.search(s)
                    if mth:
                        last["material"] = mth.group(2).strip()
                if not last.get("price"):
                    ph = PRICE_HINT.search(s)
                    if ph:
                        last["price"] = ph.group(2).strip()
    return items

# =====================================================
# === ML Hook 1: Entity Extraction from Text Blobs ====
# =====================================================

def ml_extract_entities_from_text(text: str) -> list[dict]:
    """
    Return list of dicts with at least:
      {category, subcategory, item_name, material, manufacturer_name, price}
    Start with a simple LLM or spaCy model; fallback to regex extractor.
    """
    try:
        # TODO: Replace with real ML extraction later
        return extract_from_text(text.splitlines())
    except Exception:
        return extract_from_text(text.splitlines())

def enrich_blobs_with_ml(blobs: list[dict]) -> list[dict]:
    """
    For non-Excel blobs, use ML entity extraction to create a small DataFrame
    so consolidate() stays unchanged.
    """
    enriched = []
    for b in blobs:
        if b.get("excel") is not None:
            enriched.append(b)
            continue
        text = b.get("text") or ""
        if text.strip():
            ml_items = ml_extract_entities_from_text(text)
            if ml_items:
                b2 = b.copy()
                b2["excel"] = pd.DataFrame(ml_items)
                enriched.append(b2)
                continue
        enriched.append(b)
    return enriched

def consolidate(blobs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a unified table with EXACTLY these 29 columns (and order):
      Category, Subcategory, MaterialGroup, Material, DimensionalStd, Item Name,
      NPS (in), Schedule/Class, Unit of Measure, Quantity, UnitWeight_kg,
      CO2_factor_kg_per_kg, CO2e_per_unit, Price_per_unit_USD, Manufacturer_Name,
      VendorKey, Country, TransportMode, Date, OTD%, QualityScore, RiskIndex,
      RiskScore, PriceScore, CarbonScore, SPI_Score, SPI_Band, SpecKey, _SourceFile

    Additionally, create alias columns used by the NLQ module:
      category, subcategory, item_name, material, manufacturer_name, price
    """
    TARGETS = [
        "Category","Subcategory","MaterialGroup","Material","DimensionalStd","Item Name",
        "NPS (in)","Schedule/Class","Unit of Measure","Quantity","UnitWeight_kg",
        "CO2_factor_kg_per_kg","CO2e_per_unit","Price_per_unit_USD","Manufacturer_Name",
        "VendorKey","Country","TransportMode","Date","OTD%","QualityScore","RiskIndex",
        "RiskScore","PriceScore","CarbonScore","SPI_Score","SPI_Band","SpecKey","_SourceFile"
    ]

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    SYN = {
        "Category": {"category","cat"},
        "Subcategory": {"subcategory","subcat","sub_category"},
        "MaterialGroup": {"materialgroup","material_group","matgroup","matgrp","groupcode"},
        "Material": {"material","moc","materialofconstruction","grade","spec","specification"},
        "DimensionalStd": {"dimensionalstd","dimensional_std","dimstd","standard","std"},
        "Item Name": {"itemname","item","description","materialdescription","shorttext","longtext","productname"},
        "NPS (in)": {"nps","npsin","sizein","nominalpipesize"},
        "Schedule/Class": {"scheduleclass","schedule","class","sch","rating","pressureclass"},
        "Unit of Measure": {"unitofmeasure","uom","unit","units","measure"},
        "Quantity": {"quantity","qty","amount"},
        "UnitWeight_kg": {"unitweightkg","unitweight","unitweight_kg","weightkg","weighteachkg"},
        "CO2_factor_kg_per_kg": {"co2factorkgperkg","co2kgkg","emissionfactor","co2_factor"},
        "CO2e_per_unit": {"co2eperunit","co2eunit","kgco2eperunit","emissionsperunit"},
        "Price_per_unit_USD": {"priceperunitusd","unitpriceusd","unitprice","priceusd","price_per_unit","price"},
        "Manufacturer_Name": {"manufacturername","manufacturer","make","brand","vendorname"},
        "VendorKey": {"vendorkey","vendorid","supplierid"},
        "Country": {"country","origin","countryoforigin"},
        "TransportMode": {"transportmode","logisticsmode","shippingmode","modeoftransport"},
        "Date": {"date","docdate","pricedate","podate"},
        "OTD%": {"otd","otdpercent","ontimedelivery"},
        "QualityScore": {"qualityscore","qualityindex","qscore"},
        "RiskIndex": {"riskindex","riskidx"},
        "RiskScore": {"riskscore","rscore"},
        "PriceScore": {"pricescore","pscore"},
        "CarbonScore": {"carbonscore","cscore"},
        "SPI_Score": {"spiscore","spi_score"},
        "SPI_Band": {"spiband","spi_band"},
        "SpecKey": {"speckey","spec_key"},
        "_SourceFile": {"_sourcefile","sourcefile","source_file"},
    }

    REV = {}
    for can, variants in SYN.items():
        for v in variants | {_norm(can)}:
            REV[v] = can

    rows: List[Dict[str, Any]] = []

    for b in blobs:
        srcname = b.get("name","")

        # Excel path
        if isinstance(b.get("excel"), pd.DataFrame) and not b["excel"].empty:
            df = b["excel"].copy()

            # Build normalized mapping canonical->original
            colmap: Dict[str, str] = {}
            for c in df.columns:
                nc = _norm(c)
                if nc in REV:
                    colmap.setdefault(REV[nc], c)

            # If the sheet already uses exact headers, just reorder and add _SourceFile
            if all(c in df.columns for c in TARGETS if c != "_SourceFile"):
                out = pd.DataFrame()
                for tgt in TARGETS:
                    if tgt == "_SourceFile":
                        out[tgt] = srcname
                    elif tgt in df.columns:
                        out[tgt] = df[tgt]
                    elif tgt in colmap:
                        out[tgt] = df[colmap[tgt]]
                    else:
                        out[tgt] = ""
                rows.extend(out.to_dict(orient="records"))
                continue

            # Otherwise map synonyms; blanks for missing
            out = pd.DataFrame()
            for tgt in TARGETS:
                if tgt == "_SourceFile":
                    out[tgt] = srcname
                elif tgt in colmap:
                    out[tgt] = df[colmap[tgt]]
                elif tgt in df.columns:
                    out[tgt] = df[tgt]
                else:
                    out[tgt] = ""
            rows.extend(out.to_dict(orient="records"))
            continue

        # Text/other path (minimal extraction to canonical)
        text_lines = (b.get("text") or "").splitlines()
        for rec in extract_from_text(text_lines):
            row = {k:"" for k in TARGETS}
            row["Category"] = rec.get("category","")
            row["Subcategory"] = rec.get("subcategory","")
            row["Item Name"] = rec.get("item_name","")
            row["Material"] = rec.get("material","")
            row["Manufacturer_Name"] = rec.get("manufacturer_name","")
            row["Price_per_unit_USD"] = rec.get("price","")
            row["_SourceFile"] = srcname
            rows.append(row)

    # Final DF with all 29 columns in order
    if not rows:
        tbl = pd.DataFrame(columns=TARGETS)
    else:
        tbl = pd.DataFrame(rows)

    # Ensure all canonical columns exist and order them
    for col in TARGETS:
        if col not in tbl.columns:
            tbl[col] = ""
    tbl = tbl[TARGETS].fillna("").replace("nan","", regex=False).reset_index(drop=True)

    # ---- Create NLQ alias columns expected elsewhere in your app ----
    # (no renames; just add additional lowercase copies)
    try:
        tbl["category"] = tbl["Category"]
        tbl["subcategory"] = tbl["Subcategory"]
        tbl["item_name"] = tbl["Item Name"]
        tbl["material"] = tbl["Material"]
        tbl["manufacturer_name"] = tbl["Manufacturer_Name"]
        # many NLQ routines look for a generic 'price' column
        tbl["price"] = tbl["Price_per_unit_USD"]
    except Exception:
        pass

    # ---- Conservative de-dup: only drop exact duplicate rows ----
    tbl = tbl.drop_duplicates().reset_index(drop=True)

    return tbl

# =====================================================
# === ML Hook 2: Semantic Retrieval Index for NLQ ====
# =====================================================

def _semantic_text(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["item_name","material","manufacturer_name","category","subcategory"] if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df))
    return df[cols].astype(str).agg(" | ".join, axis=1)

def build_vector_index(df: pd.DataFrame):
    """Build TF-IDF index for semantic search (upgrade to embeddings later)."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel
    except Exception:
        st.warning("TF-IDF not available; semantic retrieval disabled.")
        return

    s = _semantic_text(df).fillna("")
    vect = TfidfVectorizer(min_df=2, ngram_range=(1,2))
    X = vect.fit_transform(s)
    st.session_state["_vec"] = {"vect": vect, "X": X}

def semantic_filter(df: pd.DataFrame, query: str, top_k: int = 500) -> pd.DataFrame:
    v = st.session_state.get("_vec")
    q = (query or "").strip().lower()
    if not v or not q:
        return df

    # Bypass shortlist for very broad / single-word scopes so we don't truncate counts
    BROAD = {"drilling", "piping", "structural", "welding"}
    if q in BROAD or len(q.split()) <= 1:
        return df

    from sklearn.metrics.pairwise import linear_kernel
    vect, X = v["vect"], v["X"]
    qv = vect.transform([q])
    sims = linear_kernel(qv, X).ravel()
    if sims.max() <= 0:
        return df
    top_idx = np.argsort(-sims)[: min(top_k, len(df))]
    return df.iloc[top_idx]

# =====================================================
# === ML Hook 3: Price Band Model (Stub) ============
# =====================================================
def price_band_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add p10/p50/p90 price bands and anomaly flag.
    Later: train LightGBM using material, category, vendor, country, date.
    """
    out = df.copy()
    if "price" in out.columns:
        p = pd.to_numeric(out["price"], errors="coerce")
        out["price_p50"] = p
        out["price_p10"] = p * 0.8
        out["price_p90"] = p * 1.2
        out["price_anomaly"] = (p < out["price_p10"]) | (p > out["price_p90"])
    return out


# ---------------- Upload area ----------------
st.markdown("<div class='centered-area'>", unsafe_allow_html=True)

# Always start fresh — selector removed
upload_mode = "Start fresh (Replace mode)"
st.caption("⚡ Fresh start — previous uploads are cleared each time.")

# Optional: CO₂ factors file uploader (separate)
with st.expander("Optional: Upload a CO₂ factors sheet (columns: material, co2_factor_kg_per_kg)"):
    co2_up = st.file_uploader("CO₂ factors file (XLSX/CSV)", type=["xlsx","csv"], accept_multiple_files=False, key="co2u")
    if co2_up:
        try:
            if co2_up.name.lower().endswith(".csv"):
                st.session_state.co2_factors_df = pd.read_csv(co2_up)
            else:
                st.session_state.co2_factors_df = pd.read_excel(co2_up)
            st.success(f"Loaded CO₂ table: {len(st.session_state.co2_factors_df):,} rows")
        except Exception as e:
            st.error(f"Failed to read CO₂ table: {e}")

c1, c2 = st.columns([1,1])
with c1:
    uploaded = st.file_uploader(
        "Upload any document (PDF, XLSX, JPEG, PNG, CSV, TXT…)",
        type=None, accept_multiple_files=True
    )
with c2:
    st.markdown("**Tip:** You can also ask questions below (e.g., *“What drilling item is the cheapest?”*).")

# Handle new uploads
if uploaded:
    if upload_mode == "Start fresh (Replace mode)":
        st.session_state.files = []
        st.session_state.messages = []
        for up in uploaded:
            st.session_state.files.append({"name": up.name, "bytes": up.read(), "type": up.type or "application/octet-stream"})
        st.session_state.messages.append(("assistant", "Files received. I’ll parse them when you click **Run analysis**."))
    else:
        existing = {f["name"] for f in st.session_state.files}
        for up in uploaded:
            if up.name in existing:
                toast(f"Skipped duplicate: {up.name}")
                continue
            st.session_state.files.append({"name": up.name, "bytes": up.read(), "type": up.type or "application/octet-stream"})
            st.session_state.messages.append(("assistant", "Files received. I’ll parse them when you click **Run analysis**."))

# (file cards removed)

# Bottom bar (no free-text input)
run = st.button("Run analysis", use_container_width=True)

# ------------------- ANALYSIS -------------------
if run:
    blobs = []
    with st.spinner("Parsing files…"):
        for f in st.session_state.files:
            name, ftype, bts = f["name"], f["type"], f["bytes"]
            parsed = {"name": name, "type": ftype, "text": "", "excel": None}
            low = name.lower()

            if "pdf" in ftype.lower() or low.endswith(".pdf"):
                text = read_pdf_text(bts)
                if text.strip() == "" or text.startswith("[PDF read error"):
                    text = read_pdf_scanned_ocr(bts, max_pages=20)
                parsed["text"] = text

            elif "excel" in ftype.lower() or low.endswith((".xlsx",".xlsm",".xls")):
                parsed["excel"] = read_excel_all_sheets(bts)

            elif ftype.startswith("image") or low.endswith((".png",".jpg",".jpeg",".tif",".tiff")):
                parsed["text"] = read_image_ocr(bts)

            elif low.endswith(".csv"):
                try:
                    parsed["excel"] = pd.read_csv(io.BytesIO(bts))
                except Exception:
                    parsed["text"] = bts.decode("utf-8", errors="ignore")
            elif low.endswith(".txt"):
                parsed["text"] = bts.decode("utf-8", errors="ignore")
            else:
                parsed["text"] = decode_binary(bts)

            blobs.append(parsed)

    # === Insert ML enrichment before consolidation ===
    blobs = enrich_blobs_with_ml(blobs)

    items_df = consolidate(blobs)
    items_df = price_band_predict(items_df)  # optional stub

    # (No operator multiplier applied)
    st.session_state.analysis["items"] = items_df
    st.session_state.messages.append(("assistant", "✅ File processed successfully."))
    st.session_state.messages.append(("assistant", f"💡 Found **{len(items_df)}** rows."))

# ------------------- RESULTS AREA -------------------
items_df = st.session_state.analysis.get("items", pd.DataFrame())

if not items_df.empty:
    build_vector_index(items_df)
    st.subheader("AI Analysis — Extracted Equipment Table")

    # Quick filter removed — show full table only
    view = items_df.copy()
    st.dataframe(view, use_container_width=True, height=420)

    # Exports
    c1, c2 = st.columns(2)
    with c1:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as xw:
            view.to_excel(xw, index=False, sheet_name="items")
        st.download_button(
            "⬇️ Download Excel",
            out.getvalue(),
            file_name="zproc_ai_items.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with c2:
        st.download_button(
            "⬇️ Download CSV",
            view.to_csv(index=False).encode("utf-8"),
            file_name="zproc_ai_items.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ===================== Natural-Language Query Module (v3) =====================
    import re
    from typing import Optional

    def _price_series(df: pd.DataFrame) -> pd.Series:
        cands = []
        if "avg_price_usd" in df.columns:
            cands.append(pd.to_numeric(df["avg_price_usd"], errors="coerce"))
        if {"price_low","price_high"}.issubset(df.columns):
            mid = (pd.to_numeric(df["price_low"], errors="coerce")
                   + pd.to_numeric(df["price_high"], errors="coerce")) / 2
            cands.append(mid)
        if "price" in df.columns:
            cands.append(pd.to_numeric(df["price"], errors="coerce"))
        for s in cands:
            if s.notna().sum() > 0:
                return s
        return pd.Series([], dtype=float)

    # --- NEW: weight helpers/intents ---
    def _weight_series(df: pd.DataFrame) -> pd.Series:
        """Return a numeric UnitWeight_kg series if available."""
        cands = []
        if "UnitWeight_kg" in df.columns:
            cands.append(pd.to_numeric(df["UnitWeight_kg"], errors="coerce"))
        if "unitweight_kg" in df.columns:
            cands.append(pd.to_numeric(df["unitweight_kg"], errors="coerce"))
        if "weight" in df.columns:
            cands.append(pd.to_numeric(df["weight"], errors="coerce"))
        for s in cands:
            if s.notna().sum() > 0:
                return s
        return pd.Series([], dtype=float)

    # --------- INSERTED HELPERS (below _weight_series) ----------
    def _parse_fraction_token(num: str, num2: Optional[str], den2: Optional[str]) -> float:
        """Handle tokens like 13, 13-3/8, 13 3/8."""
        base = float(num)
        if num2 and den2:
            try:
                base += float(num2) / float(den2)
            except Exception:
                pass
        return base

    def _nps_from_text_col(s: pd.Series) -> pd.Series:
        """
        Extract NPS from free text like: 'casing head 13-3/8"', 'BOP 9 5/8 in', 'size 7"'.
        Returns floats in inches where possible.
        """
        s = s.astype(str).str.lower()
        # patterns: 13-3/8", 13 3/8 in, 9-5/8, 7", 7 in
        pat_frac = re.compile(r'\b(\d{1,2})[ -](\d)\/(\d)\b')         # 13 3/8  or 13-3/8
        pat_simple_in = re.compile(r'\b(\d{1,2}(?:\.\d+)?)\s*(?:in|")\b')  # 7", 7 in, 13.375 in

        out = pd.Series(np.nan, index=s.index, dtype=float)

        # 1) 13-3/8 or 13 3/8 (optionally followed by " or in)
        m = s.str.extract(pat_frac)
        has = m.notna().all(axis=1)
        if has.any():
            out.loc[has] = m[0][has].astype(float).values + (m[1][has].astype(float) / m[2][has].astype(float)).values

        # 2) 7", 7 in, 13.375 in (don’t overwrite fraction hits)
        m2 = s.str.extract(pat_simple_in)
        has2 = m2[0].notna() & out.isna()
        if has2.any():
            out.loc[has2] = m2[0][has2].astype(float).values

        return out

    def _nps_series(df: pd.DataFrame) -> pd.Series:
        """
        Prefer the dedicated numeric NPS column; else extract from text fields.
        """
        # If you have a numeric NPS column already
        if "nps_in" in df.columns:
            s = pd.to_numeric(df["nps_in"], errors="coerce")
            if s.notna().sum() > 0:
                return s

        # Otherwise, try to mine it from these text columns
        cols = [c for c in ["Item Name","item_name","Material","material","DimensionalStd","Schedule/Class","SpecKey"] if c in df.columns]
        if not cols:
            return pd.Series([], dtype=float)

        acc = pd.Series(np.nan, index=df.index, dtype=float)
        for c in cols:
            cand = _nps_from_text_col(df[c])
            acc = acc.fillna(cand)
        return acc
    # --------- END INSERTED HELPERS ----------

    # --- keyword + phrase hybrid (already patched earlier) ---
    def _apply_keyword_filter(df: pd.DataFrame, kw: str) -> pd.DataFrame:
        """
        Hybrid keyword filter with domain synonyms:
          1) Phrase-first search across preferred columns.
          2) Token AND across preferred columns using simple stemming + synonyms.
          3) Token OR fallback across all text columns.
        Also pins category words to Category and keeps 'equipment' countable (not a stopword).
        """
        if not kw or df.empty:
            return df

        def _norm_series(s: pd.Series) -> pd.Series:
            return s.astype(str).str.lower().str.replace(r"\s+", " ", regex=True)

        # columns to consider
        text_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
        preferred = [c for c in [
            "category","subcategory","item_name","manufacturer_name","material",
            "DimensionalStd","Schedule/Class","SpecKey"
        ] if c in df.columns]
        if not text_cols:
            return df

        q_raw = kw.strip()
        q = q_raw.lower()

        # Treat whole query as a phrase first (allowing spaces, slashes, hyphens)
        phrase_pat = re.sub(r"\s+", r"\\s+", re.escape(q))
        phrase_regex = rf"\b{phrase_pat}\b"
        def _phrase_hits(cols: List[str]) -> pd.Series:
            m = pd.Series(False, index=df.index)
            for c in cols:
                m = m | _norm_series(df[c]).str.contains(phrase_regex, na=False, regex=True)
            return m

        # Pin high-level category if present
        cat_words = {"piping","drilling","structural","welding"}
        pinned_cat = None
        for t in re.findall(r"[a-z0-9]+", q):
            if t in cat_words and "category" in df.columns:
                pinned_cat = t
                break

        mask = pd.Series(True, index=df.index)
        if pinned_cat:
            cat_pat = rf"\b{re.escape(pinned_cat)}\b"
            cat_mask = _norm_series(df["category"]).str.contains(cat_pat, na=False, regex=True)
            mask &= cat_mask

        # 1) Phrase-first pass
        m_phrase = _phrase_hits(preferred if preferred else text_cols)
        if (mask & m_phrase).any():
            return df[mask & m_phrase]

        # 2) Token AND with simple stemming + domain synonyms
        tokens = re.findall(r"[a-z0-9]+", q)
        # keep 'equipment' and 'tools' countable (NOT stopwords)
        STOP = {"the","a","an","of","in","for","to","and","or","on","with",
                "item","items","material","materials","service","services"}
        toks = [t for t in tokens if t not in STOP]
        if not toks:
            return df[mask]

        # synonym helper
        def _variants(t: str) -> List[str]:
            bases = {t}
            # light stemming
            if t.endswith("ing"): bases.add(t[:-3])
            if t.endswith("es"):  bases.add(t[:-2])
            if t.endswith("s"):   bases.add(t[:-1])

            # generic synonyms
            if t in ("manu","manufacturer"):
                bases |= {"manufacturer","manufacture","manuf","oem"}
            if t == "api":
                bases.add("api")

            # domain boost: when pinned to drilling, expand 'tools' and 'mud'
            if pinned_cat == "drilling":
                if t in {"tool","tools"}:
                    bases |= {
                        # common drilling tool families / items
                        "stabilizer","reamer","bit","bits","float collar","float shoe",
                        "centralizer","sub","subs","crossover","jar","shock sub","rotary",
                        "drill pipe","dp","kelly","packer","hanger","liner hanger","plug",
                        "cementing","float equipment","shoe track","guide shoe"
                    }
                if t in {"mud","fluid","fluids"}:
                    bases |= {
                        "drilling fluid","drilling fluids","wbm","obm","base oil",
                        "water based mud","oil based mud","bentonite","barite",
                        "emulsifier","fluid loss","viscosifier","lignite","brine"
                    }
            return list(bases)

        def _token_and(cols: List[str], toks_left: List[str]) -> pd.Series:
            m = pd.Series(True, index=df.index)
            for t in toks_left:
                any_col = pd.Series(False, index=df.index)
                for v in _variants(t):
                    pat = rf"\b{re.escape(v)}\b"
                    this_tok = pd.Series(False, index=df.index)
                    for c in cols:
                        this_tok = this_tok | _norm_series(df[c]).str.contains(pat, na=False, regex=True)
                    any_col = any_col | this_tok
                m = m & any_col
            return m

        m_pref = _token_and(preferred if preferred else text_cols, toks)
        if (mask & m_pref).any():
            return df[mask & m_pref]

        # 3) Token OR fallback across all text columns (last resort)
        def _token_or(cols: List[str], toks_left: List[str]) -> pd.Series:
            m = pd.Series(False, index=df.index)
            for t in toks_left:
                for v in _variants(t):
                    pat = rf"\b{re.escape(v)}\b"
                    for c in cols:
                        m = m | _norm_series(df[c]).str.contains(pat, na=False, regex=True)
            return m

        m_all = _token_or(text_cols, toks)
        return df[mask & m_all]

    # --- helper: build an effective scope from the raw query when needed ---
    def _effective_scope_from_query(raw_q: str, inferred: str) -> str:
        q = (raw_q or "").strip().lower()
        if inferred and len(inferred.split()) >= 2:
            return inferred
        # strip leading intent words (how many / count of / total spend for/in/of …)
        q = re.sub(r"^\s*(how\s+many|count(?:\s+of)?)\s+", "", q)
        q = re.sub(r"^\s*(total\s+spend|total\s+price|sum\s+price)\s+(?:for|in|of)\s+", "", q)
        q = re.sub(r"^\s*(total\s+spend|total\s+price|sum\s+price)\s+", "", q)
        return q.strip() or inferred

    # ITEM 3 (replaced): infer scope from query with word-boundaries and min length
    def _infer_scope_from_query(q: str, df: pd.DataFrame) -> str:
        ql_raw = q.lower()
        ql = " " + re.sub(r"[^a-z0-9]+", " ", ql_raw) + " "

        for cat_word in ["piping", "drilling", "structural"]:
            if re.search(rf"\b{cat_word}\b", ql):
                return cat_word

        candidates = set()
        for col in ["category","subcategory","item_name","manufacturer_name","material"]:
            if col in df.columns:
                vals = df[col].dropna().astype(str).str.lower().str.strip().unique().tolist()
                vals = vals[:5000]
                for v in vals:
                    vnorm = re.sub(r"[^a-z0-9]+", " ", v).strip()
                    if len(vnorm) < 3:
                        continue
                    if re.search(rf"\b{re.escape(vnorm)}\b", ql):
                        candidates.add(v)

        if not candidates:
            return ""
        return max(candidates, key=lambda s: len(str(s)))

    def _top_vendors(df: pd.DataFrame, n=5, scope_kw: str="") -> pd.DataFrame:
        dd = semantic_filter(df, scope_kw)
        dd = _apply_keyword_filter(dd, scope_kw)
        if "manufacturer_name" not in dd.columns:
            return pd.DataFrame()
        price = _price_series(dd)
        if price.empty:
            return (dd.groupby("manufacturer_name").size()
                    .sort_values(ascending=False).head(n)
                    .rename("count").reset_index())
        out = (dd.assign(price_numeric=price)
                 .groupby("manufacturer_name")
                 .agg(count=("manufacturer_name","size"),
                      spend=("price_numeric","sum"))
                 .sort_values(["spend","count"], ascending=False)
                 .head(n).reset_index())
        out["spend"] = out["spend"].round(2)
        return out

    def _avg_co2(df: pd.DataFrame, co2_df: Optional[pd.DataFrame], scope_kw: str="") -> Optional[float]:
        if co2_df is None or co2_df.empty:
            return None
        dd = semantic_filter(df, scope_kw)
        dd = _apply_keyword_filter(dd, scope_kw)
        left_key = "co2_material_hint" if "co2_material_hint" in dd.columns \
                   else ("material" if "material" in dd.columns else None)
        if not left_key or "material" not in co2_df.columns:
            return None
        jj = dd.merge(co2_df[["material","co2_factor_kg_per_kg"]],
                      left_on=left_key, right_on="material", how="left")
        if jj["co2_factor_kg_per_kg"].notna().sum() == 0:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import linear_kernel
                v = TfidfVectorizer(min_df=1).fit(co2_df["material"].astype(str))
                A = v.transform(co2_df["material"].astype(str))
                B = v.transform(dd["material"].astype(str).fillna(""))
                sims = linear_kernel(B, A)
                best = sims.argmax(axis=1)
                factors = co2_df["co2_factor_kg_per_kg"].reset_index(drop=True)
                est = factors.iloc[best].dropna()
                return float(est.mean()) if not est.empty else None
            except Exception:
                return None
        return float(jj["co2_factor_kg_per_kg"].mean())

    def _manufacturers_for_keyword(df: pd.DataFrame, kw: str) -> pd.DataFrame:
        if not kw or "manufacturer_name" not in df.columns:
            return pd.DataFrame()
        fields = [c for c in ["item_name","category","subcategory","material"] if c in df.columns]
        if not fields:
            return pd.DataFrame()
        kwl = kw.strip().lower()
        mask = False
        for c in fields:
            mask = mask | df[c].astype(str).str.lower().str.contains(kwl, na=False)
        dd = df[mask]
        if dd.empty:
            return pd.DataFrame()
        out = dd["manufacturer_name"].astype(str).str.strip()
        out = out[out != ""]
        if out.empty:
            return pd.DataFrame()
        return (out.to_frame(name="manufacturer_name")
                  .groupby("manufacturer_name").size()
                  .sort_values(ascending=False)
                  .rename("count").reset_index())

    # ======== REPLACED: _ensure_nlq_columns with your version ========
    def _ensure_nlq_columns(df: pd.DataFrame) -> pd.DataFrame:
        alias_map = {
            "category": ["category", "Category"],
            "subcategory": ["subcategory", "Subcategory"],
            "item_name": ["item_name", "Item Name", "Description"],
            "manufacturer_name": ["manufacturer_name", "Manufacturer_Name", "Manufacturer Name"],
            "material": ["material", "Material"],
            "country": ["country", "Country", "Country of Origin", "Origin"],
            "price": ["price", "Price_per_unit_USD", "Unit Price", "Price USD"],
            "price_low": ["price_low", "Price Low"],
            "price_high": ["price_high", "Price High"],
            "avg_price_usd": ["avg_price_usd", "Avg Price USD"],
            # NEW: weight + NPS aliases
            "unitweight_kg": ["unitweight_kg", "UnitWeight_kg", "Weight_kg", "Unit Weight", "Unit Weight (kg)", "Weight (kg)"],
            "nps_in": ["nps_in", "NPS (in)", "NPS", "Size (in)", "Nominal Pipe Size", "Nominal Size"],
        }
        df = df.copy()
        for tgt, candidates in alias_map.items():
            for c in candidates:
                if c in df.columns:
                    df[tgt] = df[c]
                    break
        return df
    # ================================================================

    def answer_query(query: str, df: pd.DataFrame, co2_df: Optional[pd.DataFrame] = None):
        df = _ensure_nlq_columns(df)

        q = (query or "").strip().lower()
        if not q:
            st.warning("Try: **how many drilling items**, **cheapest valve**, **top 5 vendors for drilling**, **most expensive wellhead**, **total spend cameron**, **average CO₂ for valves**, **who is the manufacturer for float collar**, **country for stabilizer**.")
            return

        m_top = re.search(r"\btop\s+(\d+)\b", q)
        top_n = int(m_top.group(1)) if m_top else 5

        m_scope = re.search(r"\b(?:for|in|of)\s+(.+)$", q)
        scope_kw = m_scope.group(1).strip() if m_scope else ""
        if not scope_kw:
            scope_kw = _infer_scope_from_query(q, df)

        # --- NEW: total weight intent ---
        if "total weight" in q or re.search(r"\bsum\s+weight\b", q):
            dd = _apply_keyword_filter(df, scope_kw)
            w = _weight_series(dd)
            if w.empty or w.notna().sum() == 0:
                st.warning("No weight data available.")
                return
            qty = pd.to_numeric(dd["Quantity"], errors="coerce").fillna(1) if "Quantity" in dd.columns else 1
            total_w = float((w.fillna(0) * qty).sum())
            st.info(f"**Total weight{f' for {scope_kw}' if scope_kw else ''}:** {total_w:,.2f} kg")
            return

        # --- NEW: average weight intent ---
        if "weight" in q:
            dd = _apply_keyword_filter(df, scope_kw)
            w = _weight_series(dd)
            if w.empty or w.notna().sum() == 0:
                st.warning("No weight data available.")
            else:
                st.info(f"**Average unit weight{f' for {scope_kw}' if scope_kw else ''}:** {w.mean():,.2f} kg")
            return

        # --- SIZE / NPS intent (with optional manufacturers) ---
        if ("nps" in q) or re.search(r"\b(size|diameter|nominal\s*size)\b", q):
            kw = scope_kw  # already inferred above
            dd = _apply_keyword_filter(df, kw)
            if dd.empty:
                st.warning("No matching rows for that scope.")
                return

            # Build a clean NPS series
            nps = _nps_series(dd)
            if nps.empty or nps.notna().sum() == 0:
                st.warning("No NPS/size information found in the table or item text.")
                return

            clean = nps.dropna().round(3)

            # Sizes table: use value_counts to avoid odd 0/1 labeling
            sizes = (clean.value_counts()
                          .sort_index()
                          .rename_axis("NPS (in)")
                          .reset_index(name="count"))

            st.write(f"**Sizes for {kw or 'all items'}:**")
            st.dataframe(sizes, use_container_width=True, height=260)
            st.caption(f"Min: {clean.min():.3f} in • Avg: {clean.mean():.3f} in • Max: {clean.max():.3f} in")

            # If the user also asked for manufacturer, show a vendors table for the same scope
            if "manufacturer" in q or "vendor" in q or "supplier" in q:
                if "manufacturer_name" in dd.columns:
                    mans = (dd["manufacturer_name"].astype(str).str.strip())
                    mans = mans[mans != ""]
                    if mans.any():
                        vendors = (mans.to_frame(name="manufacturer_name")
                                      .groupby("manufacturer_name").size()
                                      .sort_values(ascending=False)
                                      .rename("count").reset_index())
                        st.write(f"**Manufacturers for {kw or 'these items'}:**")
                        st.dataframe(vendors, use_container_width=True, height=260)
            return
        # -----------------------------------

        # manufacturer intent
        manu_pat = re.search(r"(?:who\s+is\s+the\s+)?(?:manufacturer|vendor|supplier)\s+(?:for|of)\s+(.+)", q)
        if manu_pat:
            kw = manu_pat.group(1).strip()
            res = _manufacturers_for_keyword(df, kw)
            if res.empty:
                st.info(f"No manufacturers found for **{kw}**.")
            else:
                st.write(f"**Manufacturers for '{kw}':**")
                st.dataframe(res, use_container_width=True, height=240)
            return

        # COUNT — operate over full matching universe (no semantic cap)
        if re.search(r"\bhow many\b|\bcount\b", q) or re.fullmatch(r".*\s*count", q):
            kw_eff = _effective_scope_from_query(q, scope_kw)
            dd = _apply_keyword_filter(df, kw_eff)
            st.success(f"**Count:** {len(dd):,} item(s){f' matching \"{kw_eff}\"' if kw_eff else ''}.")
            return

        # average price
        if re.search(r"\baverage\b|\bavg\b", q) and "co2" not in q:
            dd = semantic_filter(df, scope_kw)
            dd = _apply_keyword_filter(dd, scope_kw)
            price = _price_series(dd)
            if price.empty or price.notna().sum() == 0:
                st.warning("No price data available to compute an average.")
            else:
                st.info(f"**Average price{f' for {scope_kw}' if scope_kw else ''}:** USD {price.mean():,.2f}")
            return

        # cheapest
        if re.search(r"\b(min|minimum|lowest|cheapest)\b(?!.*co2)", q) or re.search(r"\bcheapest\b", q):
            dd = semantic_filter(df, scope_kw)
            dd = _apply_keyword_filter(dd, scope_kw)
            price = _price_series(dd)
            if price.empty or price.notna().sum() == 0:
                st.warning("No price data available.")
            else:
                idx = price.idxmin()
                row = dd.loc[idx]
                st.info(
                    f"**Cheapest{f' in {scope_kw}' if scope_kw else ''}:** "
                    f"USD {price.min():,.2f} — {row.get('item_name','(item)')} "
                    f"({row.get('manufacturer_name','unknown')})"
                )
            return

        # most expensive
        if re.search(r"\b(max|maximum|highest|most\s+expensive)\b(?!.*co2)", q):
            dd = semantic_filter(df, scope_kw)
            dd = _apply_keyword_filter(dd, scope_kw)
            price = _price_series(dd)
            if price.empty or price.notna().sum() == 0:
                st.warning("No price data available.")
            else:
                idx = price.idxmax()
                row = dd.loc[idx]
                st.info(
                    f"**Highest{f' in {scope_kw}' if scope_kw else ''}:** "
                    f"USD {price.max():,.2f} — {row.get('item_name','(item)')} "
                    f"({row.get('manufacturer_name','unknown')})"
                )
            return

        # TOP vendors
        if re.search(r"\btop\s+\d+\s+(vendors|manufacturers)\b|\btop\s+(vendors|manufacturers)\b", q):
            out = _top_vendors(df, n=top_n, scope_kw=scope_kw)
            if out.empty:
                st.warning("No vendor data available.")
            else:
                st.write(f"**Top {top_n} vendors{f' for {scope_kw}' if scope_kw else ''}:**")
                st.dataframe(out, use_container_width=True, height=240)
            return

        # TOTAL SPEND — operate over full matching universe (no semantic cap)
        if "total spend" in q or "sum price" in q or "total price" in q or re.fullmatch(r"total\s+spend\s+.*", q):
            kw_eff = _effective_scope_from_query(q, scope_kw)
            dd = _apply_keyword_filter(df, kw_eff)
            price = _price_series(dd)
            if price.empty or price.notna().sum() == 0:
                st.warning("No price data available.")
            else:
                st.info(f"**Total spend{f' for {kw_eff}' if kw_eff else ''}:** USD {price.sum():,.2f}")
            return

        # CO2
        if "co2" in q or "carbon" in q:
            avg = _avg_co2(df, st.session_state.co2_factors_df, scope_kw)
            if avg is None:
                st.warning("CO₂ data not available or materials didn’t match the CO₂ table.")
            else:
                st.info(f"**Average CO₂ factor{f' for {scope_kw}' if scope_kw else ''}:** {avg:.2f} kgCO₂e/kg (indicative)")
            return

        st.warning(
            "I can handle: **count, average price, cheapest/most expensive, top N vendors, total spend, CO₂ average, manufacturer for <item>, country for <item>, and weight/total weight**. "
            "Try e.g. *weight for shaker screen*, *total weight for drilling tools*, *country for stabilizer*, *cheapest drilling*, *total spend cameron*."
        )

    st.subheader("Ask a question about the table")
    with st.form("nlq_form_v3", clear_on_submit=False):
        nlq = st.text_input(
            "Examples: 'what drilling item is the cheapest', 'cheapest valve', 'most expensive wellhead', 'how many valves', "
            "'top 5 vendors for drilling', 'total spend cameron', 'average CO₂ for valves', 'who is the manufacturer for float collar'"
        )
        asked = st.form_submit_button("Ask")
    if asked:
        try:
            answer_query(nlq, view, st.session_state.co2_factors_df)
        except Exception as e:
            st.error(f"Query error: {e}")
    # ===================== End NLQ v3 =====================

st.markdown("</div>", unsafe_allow_html=True)