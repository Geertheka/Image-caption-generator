import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk

for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

st.set_page_config(
    page_title="Flickr8k Caption Evaluator",
    page_icon="🖼️",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
    --bg:          #000000;
    --surface:     #111111;
    --surface2:    #1a1a1a;
    --border:      #2a2a2a;
    --border2:     #333333;
    --text:        #f0f0f0;
    --text-dim:    #888888;
    --accent:      #3b82f6;
    --accent-dark: #1d3a6e;
    --green:       #22c55e;
    --green-dark:  #14532d;
    --yellow:      #eab308;
    --yellow-dark: #713f12;
    --red:         #ef4444;
    --red-dark:    #7f1d1d;
    --radius:      12px;
    --radius-sm:   8px;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stApp"] {
    background: #000000 !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"],
#MainMenu, footer, [data-testid="stSidebar"],
[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}

[data-testid="stMainBlockContainer"] {
    max-width: 660px !important;
    margin: 0 auto !important;
    padding: 2.5rem 1.5rem 5rem !important;
}

.page-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 2rem;
    letter-spacing: -0.02em;
}

.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin: 0 0 0.5rem;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    padding: 0.25rem !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stFileUploaderDropzone"] svg {
    color: var(--text-dim) !important;
    stroke: var(--text-dim) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--text-dim) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.35rem 0.9rem !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFile"] * {
    color: var(--text-dim) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
}
[data-testid="stFileUploaderFile"] svg {
    color: var(--text-dim) !important;
    stroke: var(--text-dim) !important;
}

/* Image */
[data-testid="stImage"] { margin: 1.25rem 0 !important; }
[data-testid="stImage"] img {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border2) !important;
    width: 100% !important;
}

/* Button */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 0.75rem 1.5rem !important;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
}
[data-testid="stButton"] > button:hover { opacity: 0.88 !important; }
[data-testid="stButton"] > button:active { transform: scale(0.99); }

/* Spinner */
[data-testid="stSpinner"] * {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    color: var(--text-dim) !important;
}

/* Caption box */
.caption-box {
    padding: 1rem 1.25rem;
    background: var(--accent-dark);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    margin-bottom: 1rem;
}
.caption-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.4rem;
}
.caption-text {
    font-size: 1rem;
    color: var(--text);
    line-height: 1.6;
    margin: 0;
}

/* Reference box */
.ref-box {
    padding: 1rem 1.25rem;
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    margin-bottom: 1rem;
}
.ref-caption {
    font-size: 0.875rem;
    color: var(--text-dim);
    line-height: 1.6;
    margin: 0.3rem 0 0;
    padding-left: 0.75rem;
    border-left: 2px solid var(--border2);
}

/* BLEU score grid */
.scores-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 0.75rem 0 1rem;
}
.score-card {
    border-radius: var(--radius-sm);
    padding: 0.875rem 0.75rem;
    text-align: center;
}
.score-card.good  { background: var(--green-dark);  border: 1px solid var(--green); }
.score-card.ok    { background: var(--yellow-dark); border: 1px solid var(--yellow); }
.score-card.low   { background: var(--red-dark);    border: 1px solid var(--red); }
.score-name {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.score-card.good .score-name { color: var(--green); }
.score-card.ok   .score-name { color: var(--yellow); }
.score-card.low  .score-name { color: var(--red); }
.score-value {
    font-size: 1.5rem;
    font-weight: 600;
}
.score-card.good .score-value { color: var(--green); }
.score-card.ok   .score-value { color: var(--yellow); }
.score-card.low  .score-value { color: var(--red); }

/* Verdict badge */
.verdict {
    display: inline-block;
    padding: 0.5rem 1.1rem;
    border-radius: 999px;
    font-size: 0.875rem;
    font-weight: 600;
    margin-top: 0.25rem;
}
.verdict.excellent { background: var(--green-dark);  color: var(--green);  border: 1px solid var(--green); }
.verdict.good      { background: var(--green-dark);  color: var(--green);  border: 1px solid var(--green); }
.verdict.moderate  { background: var(--yellow-dark); color: var(--yellow); border: 1px solid var(--yellow); }
.verdict.low       { background: var(--red-dark);    color: var(--red);    border: 1px solid var(--red); }
.verdict.very-low  { background: var(--red-dark);    color: var(--red);    border: 1px solid var(--red); }

hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

[data-testid="stTextInput"] input {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border2) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
}
[data-testid="stExpander"] {
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-sm) !important;
    background: var(--surface) !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] p {
    color: var(--text-dim) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
}
</style>

<div class="page-title">Flickr8k Caption Evaluator</div>
""", unsafe_allow_html=True)


# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_model()


def generate_caption(image):
    inputs = processor(image, text="a photo of", return_tensors="pt").to(device)
    out = model.generate(
        **inputs, max_new_tokens=60, num_beams=5,
        early_stopping=True, repetition_penalty=1.3,
    )
    return processor.decode(out[0], skip_special_tokens=True)


# ── Caption file parser ────────────────────────────────────────────────────────
def parse_captions_file(content: str) -> dict:
    captions = {}
    for line in content.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                img_id = parts[0].split("#")[0].strip()
                captions.setdefault(img_id, []).append(parts[1].strip())
        elif "," in line:
            parts = line.split(",", 1)
            cap = parts[1].strip() if len(parts) > 1 else ""
            if cap:
                captions.setdefault(parts[0].strip(), []).append(cap)
        else:
            parts = line.split(None, 1)
            if len(parts) == 2:
                captions.setdefault(parts[0].strip(), []).append(parts[1].strip())
    return captions


# ── BLEU scoring ───────────────────────────────────────────────────────────────
def compute_bleu(hypothesis: str, references: list) -> dict:
    smoother = SmoothingFunction().method1
    hyp_tok  = word_tokenize(hypothesis.lower())
    ref_tok  = [word_tokenize(r.lower()) for r in references]
    return {
        "BLEU-1": sentence_bleu(ref_tok, hyp_tok, weights=(1,0,0,0),         smoothing_function=smoother),
        "BLEU-2": sentence_bleu(ref_tok, hyp_tok, weights=(.5,.5,0,0),       smoothing_function=smoother),
        "BLEU-3": sentence_bleu(ref_tok, hyp_tok, weights=(.33,.33,.33,0),   smoothing_function=smoother),
        "BLEU-4": sentence_bleu(ref_tok, hyp_tok, weights=(.25,.25,.25,.25), smoothing_function=smoother),
    }

def score_class(v):
    return "good" if v >= 0.4 else ("ok" if v >= 0.2 else "low")

def verdict(b4):
    if   b4 >= 0.40: return "excellent", "Excellent"
    elif b4 >= 0.30: return "good",      "Good"
    elif b4 >= 0.20: return "moderate",  "Moderate"
    elif b4 >= 0.10: return "low",       "Low"
    else:            return "very-low",  "Very Low"


# ── UI ─────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="section-label">Image</div>', unsafe_allow_html=True)
    img_file = st.file_uploader("img", type=["jpg","jpeg","png","webp"], key="img", label_visibility="collapsed")
with col2:
    st.markdown('<div class="section-label">Captions file</div>', unsafe_allow_html=True)
    cap_file = st.file_uploader("cap", type=["txt","token"], key="cap", label_visibility="collapsed")

img_name_override = ""
if img_file and cap_file:
    with st.expander("Override image filename for lookup"):
        img_name_override = st.text_input(
            "Exact filename as in captions file (e.g. 667626_18933d713e.jpg)",
            value="", placeholder="Leave blank to use uploaded filename"
        )

st.markdown("<hr>", unsafe_allow_html=True)

if img_file:
    image = Image.open(img_file).convert("RGB")
    st.image(image, use_container_width=True)

    btn_label = "Generate Caption & Evaluate" if cap_file else "Generate Caption"
    if st.button(btn_label):
        with st.spinner("Generating caption..."):
            caption = generate_caption(image)

        # Generated caption
        st.markdown(f"""
        <div class="caption-box">
            <div class="caption-label">Generated Caption</div>
            <p class="caption-text">{caption}</p>
        </div>
        """, unsafe_allow_html=True)

        if cap_file:
            cap_content = cap_file.read().decode("utf-8", errors="ignore")
            captions_db = parse_captions_file(cap_content)

            lookup_name = img_name_override.strip() if img_name_override.strip() else img_file.name
            references  = captions_db.get(lookup_name, [])

            if not references:
                base = os.path.splitext(lookup_name)[0].lower()
                for key in captions_db:
                    if os.path.splitext(key)[0].lower() == base:
                        references = captions_db[key]
                        break

            if references:
                # Reference captions
                refs_html = "".join(f'<p class="ref-caption">{r}</p>' for r in references)
                st.markdown(f"""
                <div class="ref-box">
                    <div class="section-label">Ground Truth ({len(references)} references)</div>
                    {refs_html}
                </div>
                """, unsafe_allow_html=True)

                # BLEU scores
                scores = compute_bleu(caption, references)
                scores.pop("BLEU-4", None)
                cards  = "".join(f"""
                    <div class="score-card {score_class(val)}">
                        <div class="score-name">{name}</div>
                        <div class="score-value">{val:.3f}</div>
                    </div>""" for name, val in scores.items())

                b4 = scores["BLEU-3"]
                vclass, vlabel = verdict(b4)

                st.markdown(f"""
                <div class="section-label" style="margin-top:1.25rem;">BLEU Scores</div>
                <div class="scores-grid">{cards}</div>
                <div class="section-label">Overall Rating</div>
                <span class="verdict {vclass}">{vlabel}</span>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="ref-box">
                    <div class="section-label">No References Found</div>
                    <p class="ref-caption">
                        Could not match <strong style="color:#f0f0f0">{lookup_name}</strong> in the captions file.
                        Use the override field to enter the exact filename.
                    </p>
                </div>
                """, unsafe_allow_html=True)