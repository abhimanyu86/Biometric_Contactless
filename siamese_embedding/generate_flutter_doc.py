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
    warn_urgent = ParagraphStyle(
        "WarnUrgent", parent=styles["Normal"], fontSize=10, leading=14,
        backColor=colors.Color(1, 0.93, 0.93), borderPadding=8,
    )

    story = []

    story.append(Paragraph("CNN/Siamese Embedding Endpoints", title_style))
    story.append(Paragraph(
        "Integration guide for the Flutter team — a second, independent matching path "
        "(single-finger and slap) alongside the existing minutiae endpoints",
        subtitle_style,
    ))

    story.append(Paragraph(
        "PLEASE READ: the slap embedding threshold was just recalibrated (0.94 -> 0.78, "
        "real-photo-based, see §7). When you test slap again, please report specifically "
        "<b>which way it fails, if it still does</b> — rejecting people who should pass "
        "(too strict) or letting the wrong person through (too loose). That direction "
        "matters more than \"it still doesn't work\" — it tells us which way to adjust next.",
        warn_urgent,
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. What this is", h2))
    story.append(Paragraph(
        "Nine new endpoints (three single-finger, three slap-embedding, three slap-hybrid) "
        "on the same cloud service you already talk to, offering alternatives to minutiae-"
        "based matching: a CNN (pretrained ResNet18) converts each fingerprint image into a "
        "512-number vector (\"embedding\"), and two images are compared by how similar their "
        "vectors are (cosine similarity), instead of comparing extracted ridge-point "
        "locations. The single-finger version is validated at 92.5% matching accuracy on a "
        "336-person test set (vs. ~52% for the existing minutiae matcher) — see the attached "
        "<i>Siamese_Embedding_Phase_A_Report.pdf</i> for the full methodology and numbers. "
        "The slap-embedding version runs that same single-finger comparison once per finger "
        "(see §4); slap-hybrid additionally combines it with the minutiae matcher per finger "
        "(see §5). Neither slap version has been separately measured for slap-specific "
        "accuracy/FAR/FRR on real multi-finger captures — treat both as functionally "
        "verified, not yet accuracy-validated at the slap level.",
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
        "default 4-finger layout.",
        body,
    ))
    story.append(Paragraph(
        "<b>Matching logic: per-finger, not averaged.</b> Each probe finger is compared "
        "only to its own same-position enrolled finger, and judged independently against "
        "its own threshold — no averaging across fingers. A candidate is accepted only if "
        "<b>at least 2 of their fingers individually clear it</b> "
        "(<code>min_fingers_required</code> below, server-configurable). A single strong "
        "finger score can no longer carry a weak one. <b>Note the threshold here is 0.78, "
        "not single-finger's 0.94</b> — recalibrated specifically for real-photo conditions "
        "(see §7); it's a different, independently-configurable number "
        "(<code>EMBEDDING_SLAP_FINGER_THRESHOLD</code>).",
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
    story.append(Paragraph(
        "\"success\" here means a candidate cleared min_fingers_required and is being "
        "returned — check fingers_passed/min_fingers_required for the actual decision detail, "
        "not a confidence-vs-threshold comparison.",
        body,
    ))
    story.append(Spacer(1, 4))
    story.append(labeled_code(
        "Response (match found):",
        '{"success": true, "uid": "...", "name": "...", "hand_side": "right", '
        '"fingers_passed": 2, "min_fingers_required": 2, "avg_confidence": 0.9957, '
        '"threshold": 0.78, "matched_fingers": [{"probe_position": "right_little", '
        '"matched_position": "right_little", "confidence": 0.9954, "passed": true}, ...]}',
        body, code,
    ))
    story.append(Spacer(1, 8))
    story.append(labeled_code(
        "Response (no match — e.g. only 1 of 2 fingers passed):",
        '{"success": false, "message": "No match found", "fingers_passed": 1, '
        '"min_fingers_required": 2, "threshold": 0.78, "matched_fingers": [...]}',
        body, code,
    ))

    story.append(Paragraph("POST /v2/verify_slap_embedding — 1:1 (confirm identity)", h3))
    story.append(Paragraph(
        "Multipart fields: <b>uid</b>, <b>batch</b>, <b>hand_side</b> (optional), "
        "<b>image</b> (multiple). Compares only against that specific person's enrolled "
        "fingers. Use <b>matched</b> for the pass/fail decision — it's already "
        "fingers_passed &gt;= min_fingers_required, no need to recompute it client-side.",
        body,
    ))
    story.append(Spacer(1, 4))
    story.append(labeled_code(
        "Response:",
        '{"success": true, "matched": true, "uid": "...", "name": "...", "hand_side": "right", '
        '"fingers_passed": 2, "min_fingers_required": 2, "avg_confidence": 0.9957, '
        '"threshold": 0.78, "matched_fingers": [{"probe_position": "right_little", '
        '"matched_position": "right_little", "confidence": 0.9954, "passed": true}, '
        '{"probe_position": "right_ring", "matched_position": "right_ring", '
        '"confidence": 0.9881, "passed": true}]}',
        body, code,
    ))

    story.append(PageBreak())

    story.append(Paragraph("5. Slap-hybrid endpoints (minutiae + embedding fused)", h2))
    story.append(Paragraph(
        "Same request shape as the slap-embedding endpoints above (multiple images, one per "
        "finger, <b>hand_side</b>/<b>finger_order</b>). The difference is what happens per "
        "finger: this path runs <b>both</b> the old minutiae matcher and the new embedding "
        "matcher on every finger, combines the two scores into one <b>fused score</b>, and "
        "judges that fused score against its own threshold. Enrollment is slower than the "
        "other two paths (it runs both extractors), but request/response shape is otherwise "
        "the same family.",
        body,
    ))
    story.append(Paragraph(
        "<b>Why fuse them at all?</b> We checked whether minutiae and embedding actually "
        "make different mistakes on the same real photos, or just fail on the same ones. They "
        "fail on partly different ones — combining them (properly weighted: 80% embedding, "
        "20% minutiae, fit by measuring real data, not guessed) measured 85.37% accuracy vs. "
        "82.73% for embedding alone on the same 30 real people — a real but modest gain, not "
        "a dramatic one. Same acceptance rule as slap-embedding: a candidate passes if at "
        "least 2 fingers individually clear the fused threshold.",
        body,
    ))

    story.append(Paragraph("POST /v2/enroll_slap_hybrid", h3))
    story.append(Paragraph("Multipart form fields: same as /v2/enroll_slap_embedding.", body))
    story.append(Spacer(1, 4))
    story.append(labeled_code(
        "Response:",
        '{"success": true, "uid": "...", "name": "...", "batch": "...", "hand_side": "right", '
        '"finger_count": 2, "enrolled_fingers": [{"finger_position": "right_little", '
        '"iso_code": 5, "minutiae_count": 83, "embedding_dim": 512}, ...]}',
        body, code,
    ))

    story.append(Paragraph("POST /v2/authenticate_slap_hybrid — 1:N (identify)", h3))
    story.append(Paragraph(
        "Multipart fields: same as /v2/authenticate_slap_embedding. Note the field is "
        "<b>fused_threshold</b>, not <b>threshold</b> — the fused score lives on a different "
        "scale (roughly -3 to +4, not 0-1) since it's a weighted combination of two "
        "z-normalized scores, not a raw similarity.",
        body,
    ))
    story.append(Spacer(1, 4))
    story.append(labeled_code(
        "Response (match found):",
        '{"success": true, "uid": "...", "name": "...", "hand_side": "right", '
        '"fingers_passed": 2, "min_fingers_required": 2, "avg_fused_score": 2.68679, '
        '"fused_threshold": 1.1162, "matched_fingers": [{"probe_position": "right_little", '
        '"matched_position": "right_little", "fused_score": 2.78227, "minutiae_score": '
        '0.12493, "embedding_score": 0.92298, "passed": true}, ...]}',
        body, code,
    ))
    story.append(Spacer(1, 8))
    story.append(labeled_code(
        "Response (no match):",
        '{"success": false, "message": "No match found", "fingers_passed": 1, '
        '"min_fingers_required": 2, "fused_threshold": 1.1162, "matched_fingers": [...]}',
        body, code,
    ))

    story.append(Paragraph("POST /v2/verify_slap_hybrid — 1:1 (confirm identity)", h3))
    story.append(Paragraph(
        "Multipart fields: same as /v2/verify_slap_embedding. Use <b>matched</b> for the "
        "pass/fail decision.",
        body,
    ))
    story.append(Spacer(1, 4))
    story.append(labeled_code(
        "Response:",
        '{"success": true, "matched": true, "uid": "...", "name": "...", "hand_side": "right", '
        '"fingers_passed": 2, "min_fingers_required": 2, "avg_fused_score": 2.68679, '
        '"fused_threshold": 1.1162, "matched_fingers": [{"probe_position": "right_little", '
        '"matched_position": "right_little", "fused_score": 2.78227, "minutiae_score": '
        '0.12493, "embedding_score": 0.92298, "passed": true}, {"probe_position": '
        '"right_ring", "matched_position": "right_ring", "fused_score": 2.59131, '
        '"minutiae_score": 0.05141, "embedding_score": 0.93943, "passed": true}]}',
        body, code,
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "A concrete example of the fusion actually helping, from our own testing: two "
        "impostor fingers scored 0.753 and 0.796 on embedding alone — both uncomfortably "
        "close to slap-embedding's 0.78 threshold, borderline false-accept risk. Minutiae's "
        "scores for those same two fingers were 0.022 and 0.0 — clearly \"different finger.\" "
        "The fused score pulled both well below the fused threshold, turning a near-miss "
        "into a clear, comfortable rejection.",
        body,
    ))

    story.append(PageBreak())

    story.append(Paragraph("6. Reading the response", h2))
    story.append(Paragraph(
        "<b>confidence</b> (and each finger's individual score, in matched_fingers) is a "
        "cosine similarity score from -1 to 1 (in practice you'll see roughly 0.85-1.0 for "
        "fingerprint photos, genuine or not — this embedding space sits in a narrow, high "
        "band, not spread across the full range).",
        body,
    ))
    story.append(Paragraph(
        "<b>Single-finger decision</b>: matched is a direct confidence-vs-threshold "
        "comparison (currently 0.94, returned in every response so the app never needs to "
        "hardcode it).",
        body,
    ))
    story.append(Paragraph(
        "<b>Slap decision</b> works differently — it's a count, not a magnitude comparison. "
        "Each finger is checked against threshold independently (matched_fingers[i].passed), "
        "and the response is only a match when fingers_passed &gt;= min_fingers_required "
        "(default 2). Don't try to compare the slap-level \"confidence\"/\"avg_confidence\" "
        "field against threshold yourself to derive a decision — it's informational (useful "
        "for showing the user a number), the actual decision is already made and returned "
        "for you.",
        body,
    ))

    story.append(Paragraph("7. Important — nothing here is fully calibrated yet", h2))
    story.append(Paragraph(
        "Single-finger's 0.94 was measured on a lab dataset (HKPU contactless fingerprint "
        "database — different camera, lighting, and preprocessing than this app's actual "
        "capture pipeline). It's a reasonable starting point, not a final production number "
        "— build against the <b>threshold</b> field in each response rather than hardcoding "
        "0.94 anywhere in the app, so a server-side recalibration doesn't require a client "
        "update.",
        warn,
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Slap-embedding's threshold (0.78) was recalibrated specifically from real, "
        "uncontrolled photos rather than the lab dataset — a better-grounded starting point "
        "than reusing 0.94, but still not measured against actual multi-finger slap captures. "
        "<b>This is why we need your test results</b> (see the box at the top of this "
        "document): please report which direction it fails, if it still does.",
        warn,
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>min_fingers_required = 2</b> (env-configurable, "
        "<code>EMBEDDING_SLAP_MIN_FINGERS</code> / <code>HYBRID_SLAP_MIN_FINGERS</code>) is a "
        "reasonable-sounding rule for both slap paths — require more than one finger to "
        "individually pass — but hasn't been measured against real slap captures either. "
        "Whether 2-of-4 is the right cutoff (vs. 1-of-4, 3-of-4) is open. Build against "
        "<b>min_fingers_required</b> in the response — don't hardcode \"2\" anywhere in the app.",
        warn,
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "The slap-hybrid fusion weights (80% embedding / 20% minutiae) and fused threshold "
        "(1.1162) were fit on the same 30-person real-photo dataset as the slap-embedding "
        "threshold — real data, not a guess, but still a single small dataset of individual "
        "photos, not actual multi-finger slap captures. Treat it the same way: a "
        "well-reasoned starting point, not a finished number.",
        warn,
    ))

    story.append(Paragraph("8. Example (curl)", h2))
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
        "Enroll (slap-embedding, 4 fingers):",
        f'curl -X POST {BASE_URL}/v2/enroll_slap_embedding \\<br/>'
        '&nbsp;&nbsp;-F "uid=U001" -F "name=Jane Doe" -F "batch=demo" -F "hand_side=right" \\<br/>'
        '&nbsp;&nbsp;-F "image=@little.jpg" -F "image=@ring.jpg" \\<br/>'
        '&nbsp;&nbsp;-F "image=@middle.jpg" -F "image=@index.jpg"',
        body, code,
    ))
    story.append(Spacer(1, 8))
    story.append(labeled_code(
        "Enroll (slap-hybrid, same fields, different endpoint):",
        f'curl -X POST {BASE_URL}/v2/enroll_slap_hybrid \\<br/>'
        '&nbsp;&nbsp;-F "uid=U001" -F "name=Jane Doe" -F "batch=demo" -F "hand_side=right" \\<br/>'
        '&nbsp;&nbsp;-F "image=@little.jpg" -F "image=@ring.jpg" \\<br/>'
        '&nbsp;&nbsp;-F "image=@middle.jpg" -F "image=@index.jpg"',
        body, code,
    ))

    story.append(Paragraph("9. Which one should the app actually call?", h2))
    story.append(Paragraph(
        "Three independent matchers are live now for slap: minutiae-only "
        "(<code>/v2/*_slap</code>), embedding-only (<code>/v2/*_slap_embedding</code>), and "
        "hybrid (<code>/v2/*_slap_hybrid</code>). None share enrollment data — enrolling in "
        "one doesn't enroll in the others, so a person needs separate enrollment calls if you "
        "want to compare all three. If you want to compare them side-by-side during "
        "evaluation, call all relevant endpoints with the same image(s) and show the results "
        "together; for a single production flow, use whichever one the team decides on after "
        "real usage data comes in (see §7) — that decision isn't made yet, so don't hardcode "
        "a permanent choice. Hybrid measured better than embedding-alone on our test data "
        "(85.37% vs. 82.73%), but the gap is modest, not dramatic, and neither has been "
        "validated on real slap captures — don't treat hybrid as automatically \"the good "
        "one\" without your own testing confirming it.",
        body,
    ))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Backend: backend/embedding_matcher.py + backend/slap_embedding_matching.py + "
        "backend/hybrid_matcher.py + backend/slap_hybrid_matching.py + backend/cloud_app.py "
        "(branch phone-spoof). Validation report: "
        "siamese_embedding/Siamese_Embedding_Phase_A_Report.pdf. Hybrid calibration: "
        "siamese_embedding/compare_matchers_30.py, fusion_params.json.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey),
    ))

    doc.build(story)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
