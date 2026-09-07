"""Builds the CNN/Siamese embedding endpoints integration doc for the Flutter team."""
import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, ListFlowable, ListItem,
    KeepTogether,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "Embedding_Endpoints_Flutter_Integration.pdf")

BASE_URL = "https://34-100-150-103.sslip.io"


def labeled_code(label_text, code_text, body_style, code_style_obj):
    """Label paragraph + code block. The code block is a single-cell Table
    (not a backColor Paragraph) because ReportLab's Paragraph borderPadding
    background miscalculates its box height when the paragraph lands as the
    first flowable on a fresh page, causing visible overlap with the previous
    page's tail content -- Tables don't have that bug. KeepTogether keeps the
    label and its code block from splitting across a page break."""
    code_table = Table(
        [[Paragraph(code_text, code_style_obj)]],
        colWidths=[6.6 * inch],
    )
    code_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0.95, 0.95, 0.95)),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return KeepTogether([
        Paragraph(label_text, body_style),
        Spacer(1, 3),
        code_table,
    ])


def code_style(styles):
    return ParagraphStyle(
        "Code", parent=styles["Normal"], fontName="Courier", fontSize=8.5, leading=12,
    )


def build():
    doc = SimpleDocTemplate(
        OUT_PATH, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=19, spaceAfter=6)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11,
                                     textColor=colors.grey, spaceAfter=18)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=16, spaceAfter=8)
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("BodyCustom", parent=styles["Normal"], fontSize=10, leading=15)
    code = code_style(styles)
    warn = ParagraphStyle(
        "Warn", parent=styles["Normal"], fontSize=9.5, leading=14,
        backColor=colors.Color(1, 0.97, 0.87), borderPadding=8,
    )

    story = []

    story.append(Paragraph("CNN/Siamese Embedding Endpoints", title_style))
    story.append(Paragraph(
        "Integration guide for the Flutter team — a second, independent matching path "
        "(single-finger and slap) alongside the existing minutiae endpoints",
        subtitle_style,
    ))

    story.append(Paragraph("1. What this is", h2))
    story.append(Paragraph(
        "Six new endpoints (three single-finger, three slap) on the same cloud service you "
        "already talk to, offering an alternative to minutiae-based matching: a CNN "
        "(pretrained ResNet18) converts each fingerprint image into a 512-number vector "
        "(\"embedding\"), and two images are compared by how similar their vectors are "
        "(cosine similarity), instead of comparing extracted ridge-point locations. The "
        "single-finger version is validated at 92.5% matching accuracy on a 336-person test "
        "set (vs. ~52% for the existing minutiae matcher) — see the attached "
        "<i>Siamese_Embedding_Phase_A_Report.pdf</i> for the full methodology and numbers. "
        "The slap version uses the exact same per-finger matcher and aggregation shape as the "
        "existing minutiae slap endpoints, but has <b>not</b> been separately measured for "
        "slap-specific accuracy/FAR/FRR — treat it as functionally verified, not yet "
        "accuracy-validated.",
        body,
    ))
    story.append(Paragraph(
        "This runs <b>alongside</b> the existing endpoints, not instead of them. Nothing about "
        "<code>/v2/enroll</code>, <code>/v2/authenticate</code>, <code>/v2/verify</code>, "
        "<code>/v2/enroll_slap</code>, <code>/v2/authenticate_slap</code>, or "
        "<code>/v2/verify_slap</code> changes — they still work exactly as before, on their "
        "own separate data tables.",
        body,
    ))

    story.append(Paragraph("2. Base URL", h2))
    story.append(Paragraph(BASE_URL, code))
    story.append(Paragraph("Same host as the existing /v2/* minutiae endpoints.", body))

    story.append(Paragraph("3. Single-finger endpoints", h2))

    story.append(Paragraph("POST /v2/enroll_embedding", h3))
    story.append(Paragraph("Multipart form fields:", body))
    enroll_rows = [
        ["Field", "Required", "Notes"],
        ["image", "Yes", "The preprocessed fingerprint image (same as you'd send to /v2/enroll)"],
        ["name", "Yes", "Person's name"],
        ["uid", "Yes", "Unique person identifier"],
        ["batch", "Yes", "Batch/group identifier — authenticate matches within one batch"],
        ["finger_position", "No", "Integer 0-10, default 0"],
    ]
    t1 = Table(enroll_rows, colWidths=[1.3 * inch, 0.9 * inch, 4 * inch])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.85, 0.87, 0.93)),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t1)
    story.append(Spacer(1, 8))
    story.append(labeled_code(
        "Response (success):", '{"success": true, "embedding_dim": 512}', body, code,
    ))
    story.append(Spacer(1, 8))
    story.append(labeled_code(
        "Response (error, e.g. missing field or bad image):",
        '{"success": false, "error": "..."}', body, code,
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("POST /v2/authenticate_embedding — 1:N (identify)", h3))
    story.append(Paragraph(
        "Multipart fields: <b>batch</b>, <b>image</b>. Compares the image against every "
        "person enrolled in that batch, returns the best match.",
        body,
    ))
    story.append(Spacer(1, 4))
    story.append(labeled_code(
        "Response (match found):",
        '{"success": true, "name": "...", "uid": "...", "confidence": 0.9954, "threshold": 0.94}',
        body, code,
    ))
    story.append(Spacer(1, 8))
    story.append(labeled_code(
        "Response (no match / below threshold):",
        '{"success": false, "message": "No match found", "confidence": 0.6123, "threshold": 0.94}',
        body, code,
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("POST /v2/verify_embedding — 1:1 (confirm identity)", h3))
    story.append(Paragraph(
        "Multipart fields: <b>uid</b>, <b>batch</b>, <b>image</b>. Compares the image only "
        "against that specific person's enrolled embedding.",
        body,
    ))
    story.append(Spacer(1, 4))
    story.append(labeled_code(
        "Response:",
        '{"success": true, "matched": true, "confidence": 0.9954, "threshold": 0.94, '
        '"name": "...", "uid": "..."}',
        body, code,
    ))
    story.append(Spacer(1, 8))
    story.append(labeled_code(
        "Response (uid not found in batch):",
        '{"success": false, "error": "No user found: &lt;uid&gt;"}  — HTTP 404',
        body, code,
    ))

    story.append(PageBreak())

    story.append(Paragraph("4. Slap endpoints", h2))
    story.append(Paragraph(
        "Same request shape as the existing minutiae slap endpoints: multiple images in the "
        "<code>image</code> field (one per finger, in capture order), plus <b>hand_side</b> "
        "(\"left\"/\"right\", default \"right\") and an optional <b>finger_order</b> "
        "(comma-separated finger-position names) if your capture order doesn't match the "
        "default 4-finger layout. Each candidate's score is the average of their top 2 "
        "per-finger cosine-similarity scores — same aggregation rule as the minutiae slap path.",
        body,
    ))

    story.append(Paragraph("POST /v2/enroll_slap_embedding", h3))
    story.append(Paragraph("Multipart form fields:", body))
    slap_enroll_rows = [
        ["Field", "Required", "Notes"],
        ["image", "Yes (multiple)", "One image per finger, same field name repeated"],
        ["uid", "Yes", "Unique person identifier"],
        ["name", "No", "Person's name"],
        ["batch", "No", "Default \"default\""],
        ["hand_side", "No", "\"left\" or \"right\", default \"right\""],
        ["finger_order", "No", "Comma-separated finger-position override"],
    ]
    t2 = Table(slap_enroll_rows, colWidths=[1.3 * inch, 1.3 * inch, 3.6 * inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.85, 0.87, 0.93)),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 8))
    story.append(labeled_code(
        "Response:",
        '{"success": true, "uid": "...", "name": "...", "batch": "...", "hand_side": "right", '
        '"finger_count": 4, "enrolled_fingers": [{"finger_position": "right_little", '
        '"iso_code": 5, "embedding_dim": 512}, ...]}',
        body, code,
    ))

    story.append(Paragraph("POST /v2/authenticate_slap_embedding — 1:N (identify)", h3))
    story.append(Paragraph(
        "Multipart fields: <b>batch</b>, <b>hand_side</b> (optional), <b>image</b> "
        "(multiple). Compares against every person enrolled in the batch.",
        body,
    ))
    story.append(Spacer(1, 4))
    story.append(labeled_code(
        "Response (match found):",
        '{"success": true, "uid": "...", "name": "...", "hand_side": "right", '
        '"confidence": 0.9957, "avg_confidence": 0.9957, "threshold": 0.94, '
        '"matched_fingers": [{"probe_position": "right_little", '
        '"matched_position": "right_little", "confidence": 0.9954}, ...]}',
        body, code,
    ))
    story.append(Spacer(1, 8))
    story.append(labeled_code(
        "Response (no match):",
        '{"success": false, "message": "No match found", "confidence": 0.61, "threshold": 0.94}',
        body, code,
    ))

    story.append(Paragraph("POST /v2/verify_slap_embedding — 1:1 (confirm identity)", h3))
    story.append(Paragraph(
        "Multipart fields: <b>uid</b>, <b>batch</b>, <b>hand_side</b> (optional), "
        "<b>image</b> (multiple). Compares only against that specific person's enrolled "
        "fingers.",
        body,
    ))
    story.append(Spacer(1, 4))
    story.append(labeled_code(
        "Response:",
        '{"success": true, "matched": true, "uid": "...", "name": "...", "hand_side": "right", '
        '"confidence": 0.9957, "avg_confidence": 0.9957, "threshold": 0.94, '
        '"matched_fingers": [...]}',
        body, code,
    ))

    story.append(PageBreak())

    story.append(Paragraph("5. Reading the response", h2))
    story.append(Paragraph(
        "<b>confidence</b> is a cosine similarity score from -1 to 1 (in practice you'll see "
        "roughly 0.85-1.0 for fingerprint photos, genuine or not — this embedding space sits "
        "in a narrow, high band, not spread across the full range). <b>matched</b> / whether "
        "a candidate is returned at all is decided by comparing confidence against "
        "<b>threshold</b> (currently 0.94, also returned in every response so the app never "
        "needs to hardcode it).",
        body,
    ))

    story.append(Paragraph("6. Important — thresholds are not fully calibrated yet", h2))
    story.append(Paragraph(
        "0.94 was measured on a lab dataset (HKPU contactless fingerprint database — different "
        "camera, lighting, and preprocessing than this app's actual capture pipeline) using the "
        "<b>single-finger</b> protocol. It's a reasonable starting point, not a final production "
        "number. The slap threshold currently defaults to that same 0.94 "
        "(<code>EMBEDDING_SLAP_THRESHOLD</code>, independently configurable server-side) but "
        "has no slap-specific measurement behind it at all yet — it's a placeholder, not even a "
        "lab-validated number the way the single-finger one is. Expect both to change once "
        "they're measured against real captures from the app — build against the "
        "<b>threshold</b> field in each response rather than hardcoding 0.94 anywhere in the "
        "app, so a server-side recalibration doesn't require a client update.",
        warn,
    ))

    story.append(Paragraph("7. Example (curl)", h2))
    story.append(labeled_code(
        "Enroll (single-finger):",
        f'curl -X POST {BASE_URL}/v2/enroll_embedding \\<br/>'
        '&nbsp;&nbsp;-F "name=Jane Doe" -F "uid=U001" -F "batch=demo" \\<br/>'
        '&nbsp;&nbsp;-F "image=@fingerprint.jpg"',
        body, code,
    ))
    story.append(Spacer(1, 8))
    story.append(labeled_code(
        "Verify (single-finger):",
        f'curl -X POST {BASE_URL}/v2/verify_embedding \\<br/>'
        '&nbsp;&nbsp;-F "uid=U001" -F "batch=demo" -F "image=@fingerprint2.jpg"',
        body, code,
    ))
    story.append(Spacer(1, 8))
    story.append(labeled_code(
        "Enroll (slap, 4 fingers):",
        f'curl -X POST {BASE_URL}/v2/enroll_slap_embedding \\<br/>'
        '&nbsp;&nbsp;-F "uid=U001" -F "name=Jane Doe" -F "batch=demo" -F "hand_side=right" \\<br/>'
        '&nbsp;&nbsp;-F "image=@little.jpg" -F "image=@ring.jpg" \\<br/>'
        '&nbsp;&nbsp;-F "image=@middle.jpg" -F "image=@index.jpg"',
        body, code,
    ))

    story.append(Paragraph("8. Which one should the app actually call?", h2))
    story.append(Paragraph(
        "Both matchers are live and independent — the minutiae ones and these embedding ones "
        "don't share data, so enrolling in one doesn't enroll in the other (same rule for "
        "slap: the minutiae slap tables and the embedding slap tables are entirely separate). "
        "If you want to compare them side-by-side in the app during evaluation, call both "
        "endpoints with the same image(s) and show both results; for a single production flow, "
        "use whichever one the team decides on after the thresholds are recalibrated on real "
        "data (see §6) — that decision isn't made yet, so don't hardcode a permanent choice. "
        "For slap specifically, remember the accuracy itself hasn't been validated yet either, "
        "so this isn't ready to be the sole production path regardless of threshold tuning.",
        body,
    ))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Backend: backend/embedding_matcher.py + backend/slap_embedding_matching.py + "
        "backend/cloud_app.py (branch siamese-embedding). Validation report: "
        "siamese_embedding/Siamese_Embedding_Phase_A_Report.pdf.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey),
    ))

    doc.build(story)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
