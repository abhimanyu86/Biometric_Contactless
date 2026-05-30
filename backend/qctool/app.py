import base64
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

BACKEND_URL = "http://localhost:5001"

def b64_to_image(b64_string):

    image_bytes = base64.b64decode(b64_string)

    return Image.open(io.BytesIO(image_bytes))

st.set_page_config(
    page_title="UIDAI QC Tool",
    layout="wide"
)

import requests
import streamlit as st
from PIL import Image

tab1, tab2, tab3 = st.tabs([
    "Image Quality",
    "Contact Processing",
    "Contact vs Contactless"
])
def compute_ssim(img1, img2):

    img1 = cv2.cvtColor(
        np.array(img1),
        cv2.COLOR_RGB2GRAY
    )

    img2 = cv2.cvtColor(
        np.array(img2),
        cv2.COLOR_RGB2GRAY
    )

    img2 = cv2.resize(
        img2,
        (img1.shape[1], img1.shape[0])
    )

    score, _ = ssim(
        img1,
        img2,
        full=True
    )

    return round(score,4)
def compute_orb(img1, img2):

    img1 = cv2.cvtColor(
        np.array(img1),
        cv2.COLOR_RGB2GRAY
    )

    img2 = cv2.cvtColor(
        np.array(img2),
        cv2.COLOR_RGB2GRAY
    )

    img2 = cv2.resize(
        img2,
        (img1.shape[1], img1.shape[0])
    )

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = bf.match(des1, des2)

    return len(matches)
def compute_sift(img1, img2):

    img1 = cv2.cvtColor(
        np.array(img1),
        cv2.COLOR_RGB2GRAY
    )

    img2 = cv2.cvtColor(
        np.array(img2),
        cv2.COLOR_RGB2GRAY
    )

    img2 = cv2.resize(
        img2,
        (img1.shape[1], img1.shape[0])
    )

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    return len(good)

DEFAULT_BASE_URL = "http://127.0.0.1:5001"
TIMEOUT_SECONDS = 90
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
LOGO_PATH = REPO_ROOT / "assets" / "images" / "app_logo.png"


def load_logo_base64() -> str:
    if not LOGO_PATH.exists():
        return ""
    return base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")


def post_image(base_url: str, endpoint: str, uploaded_file):
    uploaded_file.seek(0)
    files = {
        "image": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    return requests.post(
        f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
        files=files,
        timeout=TIMEOUT_SECONDS,
    )


def decode_base64_image(data: str):
    try:
        return Image.open(BytesIO(base64.b64decode(data)))
    except Exception:
        return None


def find_base64_image(payload):
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str):
                image = decode_base64_image(value)
                if image is not None:
                    return key, image
            elif isinstance(value, dict):
                nested = find_base64_image(value)
                if nested is not None:
                    return nested
            elif isinstance(value, list):
                for item in value:
                    nested = find_base64_image(item)
                    if nested is not None:
                        return nested
    return None


def parse_json_response(response):
    try:
        payload = response.json()
    except ValueError:
        st.error(f"Backend returned non-JSON response (HTTP {response.status_code}).")
        st.code(response.text)
        return None

    if response.ok:
        return payload

    message = payload.get("error") or payload.get("message") or "Request failed"
    st.error(f"{message} (HTTP {response.status_code})")
    with st.expander("Backend response"):
        st.json(payload)
    return None


def metric_card(label: str, value: str, hint: str = ""):
    hint_html = f"<div class='metric-hint'>{hint}</div>" if hint else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def grade_colors(grade: str):
    palette = {
        "Excellent": ("#1E6F45", "#E6F6EC"),
        "Good": ("#8C5A07", "#FFF1D8"),
        "Marginal": ("#A66218", "#FFF4E8"),
        "Rejected": ("#B33A33", "#FDECEA"),
    }
    return palette.get(grade or "", ("#475467", "#F2F4F7"))


def build_breakdown_rows(breakdown: dict):
    rows = []
    if not isinstance(breakdown, dict):
        return rows

    for key, value in breakdown.items():
        try:
            rows.append(
                {
                    "Metric": key.replace("_", " ").title(),
                    "Score": float(value),
                }
            )
        except (TypeError, ValueError):
            continue
    return rows


def inject_styles():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700;800&display=swap');

            :root {
                --bg: #f7f2e8;
                --panel: rgba(255, 252, 247, 0.94);
                --panel-strong: #fffdfa;
                --stroke: rgba(184, 120, 0, 0.16);
                --stroke-strong: rgba(184, 120, 0, 0.28);
                --text: #1b1d21;
                --muted: #5b6470;
                --amber: #e0a020;
                --amber-deep: #b87800;
                --amber-soft: #fff3d7;
                --hero-shadow: 0 20px 48px rgba(110, 79, 19, 0.10);
                --card-shadow: 0 12px 32px rgba(110, 79, 19, 0.07);
                --radius-xl: 28px;
                --radius-lg: 20px;
                --radius-md: 16px;
            }

            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(255, 212, 118, 0.12), transparent 22%),
                    radial-gradient(circle at top left, rgba(255, 159, 67, 0.10), transparent 18%),
                    linear-gradient(180deg, #fffaf3 0%, #f7f2e8 44%, #f2ecdf 100%);
                color: var(--text);
                font-family: 'DM Sans', sans-serif;
            }

            .stAppViewContainer,
            [data-testid="stAppViewContainer"] {
                min-height: 100vh;
            }

            .block-container {
                max-width: 1380px;
                padding-top: 1.1rem;
                padding-bottom: 2.5rem;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }

            #MainMenu,
            footer,
            header[data-testid="stHeader"] {
                visibility: hidden;
            }

            h1, h2, h3, h4, h5, p, span, div, label {
                font-family: 'DM Sans', sans-serif !important;
            }

            .tool-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1.2rem;
                min-height: 11vh;
                padding: 1rem 1.35rem;
                margin-bottom: 1.1rem;
                border-radius: 24px;
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(255, 247, 231, 0.98));
                border: 1px solid var(--stroke);
                box-shadow: 0 16px 36px rgba(110, 79, 19, 0.08);
            }

            .brand-lockup {
                display: flex;
                align-items: center;
                gap: 0.9rem;
                min-width: 0;
            }

            .brand-logo {
                width: 54px;
                height: 54px;
                flex: 0 0 auto;
                border-radius: 14px;
                box-shadow: 0 8px 18px rgba(224, 160, 32, 0.18);
            }

            .brand-title {
                margin: 0;
                font-size: clamp(1.55rem, 2.1vw, 2.15rem);
                line-height: 1.02;
                letter-spacing: -0.03em;
                color: var(--text);
                font-weight: 800;
            }

            .brand-subtitle {
                margin-top: 0.1rem;
                color: #8a5b11;
                font-size: 0.92rem;
                font-weight: 700;
            }

            .header-side {
                text-align: right;
            }

            .panel-card {
                padding: 1.1rem 1.15rem;
                border-radius: var(--radius-lg);
                background: var(--panel);
                border: 1px solid var(--stroke);
                box-shadow: var(--card-shadow);
            }

            .panel-title {
                margin: 0 0 0.35rem;
                font-size: 0.9rem;
                color: #8a5b11;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 800;
            }

            .panel-body {
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.95rem;
            }

            .stTextInput > div > div > input {
                background: #fffdfa;
                color: var(--text);
                border: 1px solid var(--stroke-strong);
                border-radius: 14px;
            }

            .stFileUploader {
                border-radius: var(--radius-md);
            }

            [data-testid="stFileUploader"] {
                border: 1px dashed rgba(184, 120, 0, 0.32);
                border-radius: var(--radius-md);
                background: rgba(255, 252, 247, 0.82);
                padding: 0.35rem;
            }

            .stButton > button {
                width: 100%;
                border: 0;
                border-radius: 14px;
                padding: 0.82rem 1rem;
                background: linear-gradient(135deg, #d98f0f 0%, #f0b537 100%);
                color: #1b1d21;
                font-weight: 800;
                box-shadow: 0 12px 26px rgba(217, 143, 15, 0.18);
            }

            .stButton > button:hover {
                color: #1b1d21;
                border: 0;
            }

            [data-baseweb="tab-list"] {
                gap: 0.65rem;
                margin-bottom: 1rem;
                padding: 0.35rem;
                border-radius: 18px;
                background: rgba(255, 252, 247, 0.72);
                border: 1px solid rgba(184, 120, 0, 0.10);
                width: fit-content;
            }

            [data-baseweb="tab"] {
                height: 2.85rem;
                padding: 0 1rem;
                border-radius: 999px;
                background: transparent;
                border: 1px solid transparent;
                color: #7a5922 !important;
                font-weight: 700;
            }

            [data-baseweb="tab"][aria-selected="true"] {
                background: linear-gradient(135deg, rgba(224, 160, 32, 0.22), rgba(255, 241, 214, 0.98));
                border-color: rgba(184, 120, 0, 0.16);
                color: #4f3407 !important;
                box-shadow: 0 6px 14px rgba(184, 120, 0, 0.10);
            }

            [data-baseweb="tab"] p,
            [data-baseweb="tab"] span,
            [data-baseweb="tab"] div {
                color: inherit !important;
            }

            .section-card {
                padding: 1.25rem;
                border-radius: var(--radius-lg);
                background: var(--panel);
                border: 1px solid var(--stroke);
                box-shadow: var(--card-shadow);
            }

            .section-title {
                margin: 0 0 0.25rem;
                font-size: 1.15rem;
                font-weight: 800;
                color: var(--text);
            }

            .section-copy {
                margin: 0 0 1rem;
                color: var(--muted);
                line-height: 1.55;
            }

            .metric-card {
                height: 100%;
                padding: 1rem 1.05rem;
                border-radius: var(--radius-md);
                background: var(--panel-strong);
                border: 1px solid rgba(184, 120, 0, 0.12);
                box-shadow: 0 10px 20px rgba(110, 79, 19, 0.05);
            }

            .metric-label {
                color: #8a5b11;
                font-size: 0.82rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .metric-value {
                margin-top: 0.35rem;
                font-size: clamp(1.5rem, 2vw, 2.2rem);
                color: var(--text);
                font-weight: 800;
            }

            .metric-hint {
                margin-top: 0.35rem;
                color: var(--muted);
                font-size: 0.9rem;
            }

            .workflow-grid {
                display: grid;
                grid-template-columns: minmax(320px, 0.95fr) minmax(0, 1.25fr);
                gap: 1rem;
                margin-top: 1rem;
                align-items: start;
            }

            .workflow-panel {
                padding: 1.2rem;
                border-radius: var(--radius-lg);
                background: var(--panel);
                border: 1px solid var(--stroke);
                box-shadow: var(--card-shadow);
            }

            .workflow-stack {
                display: flex;
                flex-direction: column;
                gap: 0.9rem;
            }

            .workflow-panel h4 {
                margin: 0 0 0.3rem;
                color: var(--text);
                font-size: 1rem;
                font-weight: 800;
            }

            .workflow-panel p {
                margin: 0 0 0.9rem;
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.93rem;
            }

            .action-note {
                color: #86612a;
                font-size: 0.88rem;
                line-height: 1.5;
            }

            .result-frame {
                padding: 1rem;
                border-radius: var(--radius-md);
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(184, 120, 0, 0.12);
            }

            .uploader-shell {
                padding: 1rem;
                border-radius: var(--radius-md);
                background: rgba(255, 254, 250, 0.9);
                border: 1px solid rgba(184, 120, 0, 0.10);
            }

            .empty-state {
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 220px;
                padding: 1.2rem;
                border-radius: var(--radius-md);
                background: rgba(255, 253, 248, 0.92);
                border: 1px dashed rgba(184, 120, 0, 0.20);
                color: var(--muted);
                text-align: center;
                line-height: 1.6;
            }

            .app-footer {
                margin-top: auto;
                padding: 1rem 0 0.35rem;
                text-align: center;
                color: #8a6a33;
                font-size: 0.88rem;
                letter-spacing: 0.02em;
            }

            .tab-spacer {
                min-height: 38vh;
            }

            .grade-chip {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 0.55rem 0.8rem;
                font-size: 0.95rem;
                font-weight: 800;
            }

            @media (max-width: 1024px) {
                .tool-header {
                    flex-direction: column;
                    align-items: flex-start;
                }

                .workflow-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(base_url: str):
    logo_b64 = load_logo_base64()
    logo_html = ""
    if logo_b64:
        logo_html = f"<img class='brand-logo' src='data:image/png;base64,{logo_b64}' alt='YellowSense logo' />"

    st.markdown(
        f"""
        <section class="tool-header">
            <div class="brand-lockup">
                {logo_html}
                <div>
                    <h1 class="brand-title">UIDAI SITAA Desktop Tool</h1>
                    <div class="brand-subtitle">by YellowSense</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_readiness_tab(base_url: str):
    st.markdown(
        """
        <div class="section-card">
            <h3 class="section-title">Fingerprint Readiness Assessment</h3>
            <p class="section-copy">
                Upload a fingerprint image to submit it to <code>/readiness</code> and review
                the returned score, quality grade, and metric breakdown.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.92, 1.35])

    with left:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="workflow-panel">
                <h4>Upload Fingerprint Image</h4>
                <p>Select a single image for readiness scoring.</p>
                <div class="action-note">Supported formats: PNG, JPG, JPEG, BMP and WEBP.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='workflow-panel uploader-shell'>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Fingerprint image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="readiness_upload",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.markdown("<div class='result-frame'>", unsafe_allow_html=True)
            st.image(uploaded, caption="Uploaded fingerprint image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        run = st.button("Run readiness assessment", key="run_readiness")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="workflow-panel">
                <h4>Assessment Result</h4>
                <p>Readiness score, quality band and metric summary are presented in this panel.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not run:
            st.markdown(
                "<div class='workflow-panel'><div class='empty-state'>Upload a fingerprint image and run the assessment to view the result summary.</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if uploaded is None:
            st.warning("Upload a fingerprint image before running the assessment.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        with st.spinner("Submitting image to /readiness..."):
            try:
                response = post_image(base_url, "/readiness", uploaded)
            except requests.RequestException as exc:
                st.error(f"Could not reach /readiness: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

        payload = parse_json_response(response)
        if payload is None:
            st.markdown("</div>", unsafe_allow_html=True)
            return

        score = payload.get("readiness_score", payload.get("score"))
        grade = payload.get("grade", "Unknown")
        breakdown = payload.get("breakdown", {})

        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = None

        grade_fg, grade_bg = grade_colors(grade)

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card(
                "Readiness Score",
                f"{score_value:.1f}" if score_value is not None else "N/A",
                "0 to 100 scale",
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Grade</div>
                    <div style="margin-top:0.75rem;">
                        <span class="grade-chip" style="color:{grade_fg}; background:{grade_bg};">{grade}</span>
                    </div>
                    <div class="metric-hint">Returned by the readiness endpoint.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            metric_card(
                "Reported Metrics",
                str(len(breakdown) if isinstance(breakdown, dict) else 0),
                "Numeric items in breakdown",
            )

        if score_value is not None:
            st.progress(
                min(max(score_value / 100.0, 0.0), 1.0),
                text=f"Overall readiness score: {score_value:.1f}/100",
            )

        rows = build_breakdown_rows(breakdown)
        chart_col, data_col = st.columns([1.2, 1])
        with chart_col:
            st.markdown("#### Quality Breakdown")
            if rows:
                st.bar_chart(rows, x="Metric", y="Score", horizontal=True, color="#D98F0F")
            else:
                st.info("No numeric breakdown values were returned.")
        with data_col:
            st.markdown("#### Response Payload")
            st.json(payload)
        st.markdown("</div>", unsafe_allow_html=True)


def render_contact_tab(base_url: str):
    st.markdown(
        """
        <div class="section-card">
            <h3 class="section-title">Contact Image Processing</h3>
            <p class="section-copy">
                Upload a contact fingerprint image to submit it to <code>/process_contact</code>
                and review the processed output returned by the backend.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.92, 1.35])

    with left:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="workflow-panel">
                <h4>Upload Contact Image</h4>
                <p>Select a contact fingerprint image for backend preprocessing.</p>
                <div class="action-note">Use a clear single-frame image for processing and output review.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='workflow-panel uploader-shell'>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Contact fingerprint image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="contact_upload",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.markdown("<div class='result-frame'>", unsafe_allow_html=True)
            st.image(uploaded, caption="Uploaded contact image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        run = st.button("Run contact processing", key="run_contact")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="workflow-panel">
                <h4>Processing Result</h4>
                <p>The processed preview and response details are shown here after completion.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not run:
            st.markdown(
                "<div class='workflow-panel'><div class='empty-state'>Upload a contact image and run processing to view the output.</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if uploaded is None:
            st.warning("Upload a contact fingerprint image before running this workflow.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        with st.spinner("Submitting image to /process_contact..."):
            try:
                response = post_image(base_url, "/process_contact", uploaded)
            except requests.RequestException as exc:
                st.error(f"Could not reach /process_contact: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

        payload = parse_json_response(response)
        if payload is None:
            st.markdown("</div>", unsafe_allow_html=True)
            return

        found_image = find_base64_image(payload)
        preview_col, result_col = st.columns([1.2, 1])

        with preview_col:
            st.markdown("#### Processed Preview")
            if found_image is None:
                st.info("The response did not contain a decodable base64 image.")
            else:
                image_key, image = found_image
                st.image(
                    image,
                    caption=f"Decoded image from `{image_key}`",
                    use_container_width=True,
                )

        with result_col:
            st.markdown("#### Response Payload")
            st.json(payload)
        st.markdown("</div>", unsafe_allow_html=True)


def main():
    page_icon = None
    if LOGO_PATH.exists():
        page_icon = Image.open(LOGO_PATH)

    st.set_page_config(
        page_title="UIDAI SITAA Desktop Tool",
        page_icon=page_icon,
        layout="wide",
    )

    inject_styles()

    base_url = DEFAULT_BASE_URL
    render_header(base_url)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Tab 1: Image Quality",
            "Tab 2: Readiness",
            "Tab 3: Comparison",
            "Tab 4: Contact Processing",
        ]
    )

    with tab1:
        st.markdown("<div class='tab-spacer'></div>", unsafe_allow_html=True)
    with tab2:
        render_readiness_tab(base_url)
    with tab3:
        st.markdown("<div class='tab-spacer'></div>", unsafe_allow_html=True)
    with tab4:
        render_contact_tab(base_url)

    st.markdown(
        "<div class='app-footer'>YellowSense Technologies · SITAA Cohort 1 · 2025</div>",
        unsafe_allow_html=True,
    )


                st.error(data["error"])
# TAB 3
with tab3:

    st.header("Contact vs Contactless Comparison")

    contact = st.file_uploader(
        "Upload Contact Fingerprint",
        type=["png","jpg","jpeg"],
        key="contact_img"
    )

    contactless = st.file_uploader(
        "Upload Contactless Fingerprint",
        type=["png","jpg","jpeg"],
        key="contactless_img"
    )
    if contact and contactless:

        img1 = Image.open(contact)
        img2 = Image.open(contactless)

        col1,col2 = st.columns(2)

        with col1:
            st.image(img1)

        with col2:
            st.image(img2)

        if st.button("Compare"):

            score = compute_ssim(
                img1,
                img2
            )

            st.metric(
                "SSIM Score",
                score
            )
            orb_score = compute_orb(
                img1,
                img2
            )

            st.metric(
                "ORB Matches",
                orb_score
            )
            sift_score = compute_sift(
                img1,
                img2
            )

            st.metric(
                "SIFT Matches",
                sift_score
            )

import base64
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
from PIL import Image


DEFAULT_BASE_URL = "http://127.0.0.1:5001"
TIMEOUT_SECONDS = 90
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
LOGO_PATH = REPO_ROOT / "assets" / "images" / "app_logo.png"


def load_logo_base64() -> str:
    if not LOGO_PATH.exists():
        return ""
    return base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")


def post_image(base_url: str, endpoint: str, uploaded_file):
    uploaded_file.seek(0)
    files = {
        "image": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    return requests.post(
        f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
        files=files,
        timeout=TIMEOUT_SECONDS,
    )


def decode_base64_image(data: str):
    try:
        return Image.open(BytesIO(base64.b64decode(data)))
    except Exception:
        return None


def find_base64_image(payload):
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str):
                image = decode_base64_image(value)
                if image is not None:
                    return key, image
            elif isinstance(value, dict):
                nested = find_base64_image(value)
                if nested is not None:
                    return nested
            elif isinstance(value, list):
                for item in value:
                    nested = find_base64_image(item)
                    if nested is not None:
                        return nested
    return None


def parse_json_response(response):
    try:
        payload = response.json()
    except ValueError:
        st.error(f"Backend returned non-JSON response (HTTP {response.status_code}).")
        st.code(response.text)
        return None

    if response.ok:
        return payload

    message = payload.get("error") or payload.get("message") or "Request failed"
    st.error(f"{message} (HTTP {response.status_code})")
    with st.expander("Backend response"):
        st.json(payload)
    return None


def metric_card(label: str, value: str, hint: str = ""):
    hint_html = f"<div class='metric-hint'>{hint}</div>" if hint else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def grade_colors(grade: str):
    palette = {
        "Excellent": ("#1E6F45", "#E6F6EC"),
        "Good": ("#8C5A07", "#FFF1D8"),
        "Marginal": ("#A66218", "#FFF4E8"),
        "Rejected": ("#B33A33", "#FDECEA"),
    }
    return palette.get(grade or "", ("#475467", "#F2F4F7"))


def build_breakdown_rows(breakdown: dict):
    rows = []
    if not isinstance(breakdown, dict):
        return rows

    for key, value in breakdown.items():
        try:
            rows.append(
                {
                    "Metric": key.replace("_", " ").title(),
                    "Score": float(value),
                }
            )
        except (TypeError, ValueError):
            continue
    return rows


def inject_styles():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700;800&display=swap');

            :root {
                --bg: #f7f2e8;
                --panel: rgba(255, 252, 247, 0.94);
                --panel-strong: #fffdfa;
                --stroke: rgba(184, 120, 0, 0.16);
                --stroke-strong: rgba(184, 120, 0, 0.28);
                --text: #1b1d21;
                --muted: #5b6470;
                --amber: #e0a020;
                --amber-deep: #b87800;
                --amber-soft: #fff3d7;
                --hero-shadow: 0 20px 48px rgba(110, 79, 19, 0.10);
                --card-shadow: 0 12px 32px rgba(110, 79, 19, 0.07);
                --radius-xl: 28px;
                --radius-lg: 20px;
                --radius-md: 16px;
            }

            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(255, 212, 118, 0.12), transparent 22%),
                    radial-gradient(circle at top left, rgba(255, 159, 67, 0.10), transparent 18%),
                    linear-gradient(180deg, #fffaf3 0%, #f7f2e8 44%, #f2ecdf 100%);
                color: var(--text);
                font-family: 'DM Sans', sans-serif;
            }

            .stAppViewContainer,
            [data-testid="stAppViewContainer"] {
                min-height: 100vh;
            }

            .block-container {
                max-width: 1380px;
                padding-top: 1.1rem;
                padding-bottom: 2.5rem;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }

            #MainMenu,
            footer,
            header[data-testid="stHeader"] {
                visibility: hidden;
            }

            h1, h2, h3, h4, h5, p, span, div, label {
                font-family: 'DM Sans', sans-serif !important;
            }

            .tool-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1.2rem;
                min-height: 11vh;
                padding: 1rem 1.35rem;
                margin-bottom: 1.1rem;
                border-radius: 24px;
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(255, 247, 231, 0.98));
                border: 1px solid var(--stroke);
                box-shadow: 0 16px 36px rgba(110, 79, 19, 0.08);
            }

            .brand-lockup {
                display: flex;
                align-items: center;
                gap: 0.9rem;
                min-width: 0;
            }

            .brand-logo {
                width: 54px;
                height: 54px;
                flex: 0 0 auto;
                border-radius: 14px;
                box-shadow: 0 8px 18px rgba(224, 160, 32, 0.18);
            }

            .brand-title {
                margin: 0;
                font-size: clamp(1.55rem, 2.1vw, 2.15rem);
                line-height: 1.02;
                letter-spacing: -0.03em;
                color: var(--text);
                font-weight: 800;
            }

            .brand-subtitle {
                margin-top: 0.1rem;
                color: #8a5b11;
                font-size: 0.92rem;
                font-weight: 700;
            }

            .header-side {
                text-align: right;
            }

            .panel-card {
                padding: 1.1rem 1.15rem;
                border-radius: var(--radius-lg);
                background: var(--panel);
                border: 1px solid var(--stroke);
                box-shadow: var(--card-shadow);
            }

            .panel-title {
                margin: 0 0 0.35rem;
                font-size: 0.9rem;
                color: #8a5b11;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 800;
            }

            .panel-body {
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.95rem;
            }

            .stTextInput > div > div > input {
                background: #fffdfa;
                color: var(--text);
                border: 1px solid var(--stroke-strong);
                border-radius: 14px;
            }

            .stFileUploader {
                border-radius: var(--radius-md);
            }

            [data-testid="stFileUploader"] {
                border: 1px dashed rgba(184, 120, 0, 0.32);
                border-radius: var(--radius-md);
                background: rgba(255, 252, 247, 0.82);
                padding: 0.35rem;
            }

            .stButton > button {
                width: 100%;
                border: 0;
                border-radius: 14px;
                padding: 0.82rem 1rem;
                background: linear-gradient(135deg, #d98f0f 0%, #f0b537 100%);
                color: #1b1d21;
                font-weight: 800;
                box-shadow: 0 12px 26px rgba(217, 143, 15, 0.18);
            }

            .stButton > button:hover {
                color: #1b1d21;
                border: 0;
            }

            [data-baseweb="tab-list"] {
                gap: 0.65rem;
                margin-bottom: 1rem;
                padding: 0.35rem;
                border-radius: 18px;
                background: rgba(255, 252, 247, 0.72);
                border: 1px solid rgba(184, 120, 0, 0.10);
                width: fit-content;
            }

            [data-baseweb="tab"] {
                height: 2.85rem;
                padding: 0 1rem;
                border-radius: 999px;
                background: transparent;
                border: 1px solid transparent;
                color: #7a5922 !important;
                font-weight: 700;
            }

            [data-baseweb="tab"][aria-selected="true"] {
                background: linear-gradient(135deg, rgba(224, 160, 32, 0.22), rgba(255, 241, 214, 0.98));
                border-color: rgba(184, 120, 0, 0.16);
                color: #4f3407 !important;
                box-shadow: 0 6px 14px rgba(184, 120, 0, 0.10);
            }

            [data-baseweb="tab"] p,
            [data-baseweb="tab"] span,
            [data-baseweb="tab"] div {
                color: inherit !important;
            }

            .section-card {
                padding: 1.25rem;
                border-radius: var(--radius-lg);
                background: var(--panel);
                border: 1px solid var(--stroke);
                box-shadow: var(--card-shadow);
            }

            .section-title {
                margin: 0 0 0.25rem;
                font-size: 1.15rem;
                font-weight: 800;
                color: var(--text);
            }

            .section-copy {
                margin: 0 0 1rem;
                color: var(--muted);
                line-height: 1.55;
            }

            .metric-card {
                height: 100%;
                padding: 1rem 1.05rem;
                border-radius: var(--radius-md);
                background: var(--panel-strong);
                border: 1px solid rgba(184, 120, 0, 0.12);
                box-shadow: 0 10px 20px rgba(110, 79, 19, 0.05);
            }

            .metric-label {
                color: #8a5b11;
                font-size: 0.82rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .metric-value {
                margin-top: 0.35rem;
                font-size: clamp(1.5rem, 2vw, 2.2rem);
                color: var(--text);
                font-weight: 800;
            }

            .metric-hint {
                margin-top: 0.35rem;
                color: var(--muted);
                font-size: 0.9rem;
            }

            .workflow-grid {
                display: grid;
                grid-template-columns: minmax(320px, 0.95fr) minmax(0, 1.25fr);
                gap: 1rem;
                margin-top: 1rem;
                align-items: start;
            }

            .workflow-panel {
                padding: 1.2rem;
                border-radius: var(--radius-lg);
                background: var(--panel);
                border: 1px solid var(--stroke);
                box-shadow: var(--card-shadow);
            }

            .workflow-stack {
                display: flex;
                flex-direction: column;
                gap: 0.9rem;
            }

            .workflow-panel h4 {
                margin: 0 0 0.3rem;
                color: var(--text);
                font-size: 1rem;
                font-weight: 800;
            }

            .workflow-panel p {
                margin: 0 0 0.9rem;
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.93rem;
            }

            .action-note {
                color: #86612a;
                font-size: 0.88rem;
                line-height: 1.5;
            }

            .result-frame {
                padding: 1rem;
                border-radius: var(--radius-md);
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(184, 120, 0, 0.12);
            }

            .uploader-shell {
                padding: 1rem;
                border-radius: var(--radius-md);
                background: rgba(255, 254, 250, 0.9);
                border: 1px solid rgba(184, 120, 0, 0.10);
            }

            .empty-state {
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 220px;
                padding: 1.2rem;
                border-radius: var(--radius-md);
                background: rgba(255, 253, 248, 0.92);
                border: 1px dashed rgba(184, 120, 0, 0.20);
                color: var(--muted);
                text-align: center;
                line-height: 1.6;
            }

            .app-footer {
                margin-top: auto;
                padding: 1rem 0 0.35rem;
                text-align: center;
                color: #8a6a33;
                font-size: 0.88rem;
                letter-spacing: 0.02em;
            }

            .tab-spacer {
                min-height: 38vh;
            }

            .grade-chip {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 0.55rem 0.8rem;
                font-size: 0.95rem;
                font-weight: 800;
            }

            @media (max-width: 1024px) {
                .tool-header {
                    flex-direction: column;
                    align-items: flex-start;
                }

                .workflow-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(base_url: str):
    logo_b64 = load_logo_base64()
    logo_html = ""
    if logo_b64:
        logo_html = f"<img class='brand-logo' src='data:image/png;base64,{logo_b64}' alt='YellowSense logo' />"

    st.markdown(
        f"""
        <section class="tool-header">
            <div class="brand-lockup">
                {logo_html}
                <div>
                    <h1 class="brand-title">UIDAI SITAA Desktop Tool</h1>
                    <div class="brand-subtitle">by YellowSense</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_readiness_tab(base_url: str):
    st.markdown(
        """
        <div class="section-card">
            <h3 class="section-title">Fingerprint Readiness Assessment</h3>
            <p class="section-copy">
                Upload a fingerprint image to submit it to <code>/readiness</code> and review
                the returned score, quality grade, and metric breakdown.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.92, 1.35])

    with left:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="workflow-panel">
                <h4>Upload Fingerprint Image</h4>
                <p>Select a single image for readiness scoring.</p>
                <div class="action-note">Supported formats: PNG, JPG, JPEG, BMP and WEBP.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='workflow-panel uploader-shell'>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Fingerprint image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="readiness_upload",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.markdown("<div class='result-frame'>", unsafe_allow_html=True)
            st.image(uploaded, caption="Uploaded fingerprint image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        run = st.button("Run readiness assessment", key="run_readiness")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="workflow-panel">
                <h4>Assessment Result</h4>
                <p>Readiness score, quality band and metric summary are presented in this panel.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not run:
            st.markdown(
                "<div class='workflow-panel'><div class='empty-state'>Upload a fingerprint image and run the assessment to view the result summary.</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if uploaded is None:
            st.warning("Upload a fingerprint image before running the assessment.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        with st.spinner("Submitting image to /readiness..."):
            try:
                response = post_image(base_url, "/readiness", uploaded)
            except requests.RequestException as exc:
                st.error(f"Could not reach /readiness: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

        payload = parse_json_response(response)
        if payload is None:
            st.markdown("</div>", unsafe_allow_html=True)
            return

        score = payload.get("readiness_score", payload.get("score"))
        grade = payload.get("grade", "Unknown")
        breakdown = payload.get("breakdown", {})

        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = None

        grade_fg, grade_bg = grade_colors(grade)

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card(
                "Readiness Score",
                f"{score_value:.1f}" if score_value is not None else "N/A",
                "0 to 100 scale",
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Grade</div>
                    <div style="margin-top:0.75rem;">
                        <span class="grade-chip" style="color:{grade_fg}; background:{grade_bg};">{grade}</span>
                    </div>
                    <div class="metric-hint">Returned by the readiness endpoint.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            metric_card(
                "Reported Metrics",
                str(len(breakdown) if isinstance(breakdown, dict) else 0),
                "Numeric items in breakdown",
            )

        if score_value is not None:
            st.progress(
                min(max(score_value / 100.0, 0.0), 1.0),
                text=f"Overall readiness score: {score_value:.1f}/100",
            )

        rows = build_breakdown_rows(breakdown)
        chart_col, data_col = st.columns([1.2, 1])
        with chart_col:
            st.markdown("#### Quality Breakdown")
            if rows:
                st.bar_chart(rows, x="Metric", y="Score", horizontal=True, color="#D98F0F")
            else:
                st.info("No numeric breakdown values were returned.")
        with data_col:
            st.markdown("#### Response Payload")
            st.json(payload)
        st.markdown("</div>", unsafe_allow_html=True)


def render_contact_tab(base_url: str):
    st.markdown(
        """
        <div class="section-card">
            <h3 class="section-title">Contact Image Processing</h3>
            <p class="section-copy">
                Upload a contact fingerprint image to submit it to <code>/process_contact</code>
                and review the processed output returned by the backend.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.92, 1.35])

    with left:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="workflow-panel">
                <h4>Upload Contact Image</h4>
                <p>Select a contact fingerprint image for backend preprocessing.</p>
                <div class="action-note">Use a clear single-frame image for processing and output review.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='workflow-panel uploader-shell'>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Contact fingerprint image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="contact_upload",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.markdown("<div class='result-frame'>", unsafe_allow_html=True)
            st.image(uploaded, caption="Uploaded contact image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        run = st.button("Run contact processing", key="run_contact")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='workflow-stack'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="workflow-panel">
                <h4>Processing Result</h4>
                <p>The processed preview and response details are shown here after completion.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not run:
            st.markdown(
                "<div class='workflow-panel'><div class='empty-state'>Upload a contact image and run processing to view the output.</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if uploaded is None:
            st.warning("Upload a contact fingerprint image before running this workflow.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        with st.spinner("Submitting image to /process_contact..."):
            try:
                response = post_image(base_url, "/process_contact", uploaded)
            except requests.RequestException as exc:
                st.error(f"Could not reach /process_contact: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

        payload = parse_json_response(response)
        if payload is None:
            st.markdown("</div>", unsafe_allow_html=True)
            return

        found_image = find_base64_image(payload)
        preview_col, result_col = st.columns([1.2, 1])

        with preview_col:
            st.markdown("#### Processed Preview")
            if found_image is None:
                st.info("The response did not contain a decodable base64 image.")
            else:
                image_key, image = found_image
                st.image(
                    image,
                    caption=f"Decoded image from `{image_key}`",
                    use_container_width=True,
                )

        with result_col:
            st.markdown("#### Response Payload")
            st.json(payload)
        st.markdown("</div>", unsafe_allow_html=True)


def main():
    page_icon = None
    if LOGO_PATH.exists():
        page_icon = Image.open(LOGO_PATH)

    st.set_page_config(
        page_title="UIDAI SITAA Desktop Tool",
        page_icon=page_icon,
        layout="wide",
    )

    inject_styles()

    base_url = DEFAULT_BASE_URL
    render_header(base_url)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Tab 1: Image Quality",
            "Tab 2: Readiness",
            "Tab 3: Comparison",
            "Tab 4: Contact Processing",
        ]
    )

    with tab1:
        st.markdown("<div class='tab-spacer'></div>", unsafe_allow_html=True)
    with tab2:
        render_readiness_tab(base_url)
    with tab3:
        st.markdown("<div class='tab-spacer'></div>", unsafe_allow_html=True)
    with tab4:
        render_contact_tab(base_url)

    st.markdown(
        "<div class='app-footer'>YellowSense Technologies · SITAA Cohort 1 · 2025</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
