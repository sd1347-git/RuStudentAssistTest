"""
Report Generator — RBS Student Life Assistant
==============================================
Creates a polished PDF report covering:
  - Project overview and motivation
  - Every technique / tool explained in plain language
  - A dedicated section for each of the 10 visualisation charts

Run:  python generate_report.py
Output: RBS_RAG_Report.pdf
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Image, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib import colors

# ── Colour palette ─────────────────────────────────────────────────────────
RU_RED    = HexColor("#CC0033")
RU_GREY   = HexColor("#5F6A72")
RU_LIGHT  = HexColor("#F7F7F7")
RU_DARK   = HexColor("#1A1A1A")
BLUE      = HexColor("#006DB6")
ORANGE    = HexColor("#E87722")
GREEN     = HexColor("#228B22")
PURPLE    = HexColor("#7B2D8B")

VIZ_DIR   = "visualizations"
OUT_FILE  = "RBS_RAG_Report.pdf"

# ── Document setup ─────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUT_FILE,
    pagesize=letter,
    rightMargin=0.85 * inch,
    leftMargin=0.85 * inch,
    topMargin=0.9  * inch,
    bottomMargin=0.9 * inch,
)

# ── Custom styles ──────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

TITLE = S("ReportTitle",
          fontSize=26, leading=32, textColor=RU_RED,
          alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=6)

SUBTITLE = S("ReportSubtitle",
             fontSize=13, leading=18, textColor=RU_GREY,
             alignment=TA_CENTER, fontName="Helvetica", spaceAfter=4)

BYLINE = S("Byline",
           fontSize=10, leading=14, textColor=RU_GREY,
           alignment=TA_CENTER, fontName="Helvetica")

H1 = S("H1",
       fontSize=17, leading=22, textColor=RU_RED,
       fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=6,
       borderPad=4)

H2 = S("H2",
       fontSize=13, leading=18, textColor=BLUE,
       fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4)

H3 = S("H3",
       fontSize=11, leading=16, textColor=RU_GREY,
       fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=3)

BODY = S("Body",
         fontSize=10, leading=15, textColor=RU_DARK,
         fontName="Helvetica", alignment=TA_JUSTIFY, spaceAfter=6)

BULLET = S("Bullet",
           fontSize=10, leading=15, textColor=RU_DARK,
           fontName="Helvetica", leftIndent=18, bulletIndent=6,
           spaceAfter=3)

CAPTION = S("Caption",
            fontSize=8.5, leading=13, textColor=RU_GREY,
            fontName="Helvetica-Oblique", alignment=TA_CENTER,
            spaceBefore=4, spaceAfter=10)

CODE = S("Code",
         fontSize=8.5, leading=13, textColor=HexColor("#2c2c2c"),
         fontName="Courier", backColor=HexColor("#F0F0F0"),
         borderPad=6, leftIndent=12, spaceAfter=8)

CHART_TITLE = S("ChartTitle",
                fontSize=12, leading=16, textColor=RU_RED,
                fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4)

HIGHLIGHT = S("Highlight",
              fontSize=10, leading=15, textColor=BLUE,
              fontName="Helvetica-BoldOblique", spaceAfter=4)

# ── Helper builders ────────────────────────────────────────────────────────
def hr():
    return HRFlowable(width="100%", thickness=1, color=RU_RED,
                      spaceAfter=8, spaceBefore=4)

def thin_hr():
    return HRFlowable(width="100%", thickness=0.5, color=RU_GREY,
                      spaceAfter=6, spaceBefore=2)

def sp(h=8):
    return Spacer(1, h)

def body(text):
    return Paragraph(text, BODY)

def bullet(text):
    return Paragraph(f"&bull; &nbsp; {text}", BULLET)

def code(text):
    return Paragraph(text, CODE)

def img(filename, width=6.5 * inch):
    path = os.path.join(VIZ_DIR, filename)
    if not os.path.exists(path):
        return body(f"[Image not found: {filename}]")
    im = Image(path)
    aspect = im.imageHeight / float(im.imageWidth)
    im.drawWidth  = width
    im.drawHeight = width * aspect
    return im

def info_box(title, text, bg=RU_LIGHT, border=RU_RED):
    """A shaded callout box."""
    data = [[Paragraph(f"<b>{title}</b>", S("ib_t", fontSize=10,
                       fontName="Helvetica-Bold", textColor=border)),
             Paragraph(text, S("ib_b", fontSize=9.5, leading=14,
                               fontName="Helvetica", textColor=RU_DARK))]]
    t = Table(data, colWidths=[1.35 * inch, 5.15 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), bg),
        ("BOX",         (0, 0), (-1, -1), 1, border),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",  (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",(0,0), (-1,-1),  7),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",(0, 0), (-1, -1), 8),
    ]))
    return t

def two_col_table(rows, col_widths=(2.2*inch, 4.3*inch)):
    """Two-column definition/term table."""
    styled = []
    for term, defn in rows:
        styled.append([
            Paragraph(f"<b>{term}</b>", S("tc_t", fontSize=9.5,
                      fontName="Helvetica-Bold", textColor=RU_RED)),
            Paragraph(defn, S("tc_d", fontSize=9.5, leading=14,
                              fontName="Helvetica", textColor=RU_DARK))
        ])
    t = Table(styled, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), RU_LIGHT),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [HexColor("#FFFFFF"), RU_LIGHT]),
        ("BOX",          (0, 0), (-1, -1), 0.8, RU_GREY),
        ("INNERGRID",    (0, 0), (-1, -1), 0.4, RU_GREY),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    return t

# ══════════════════════════════════════════════════════════════════════════════
# BUILD CONTENT
# ══════════════════════════════════════════════════════════════════════════════
story = []

# ─────────────────────── COVER ───────────────────────────────────────────────
story += [
    sp(60),
    Paragraph("RBS Student Life Assistant", TITLE),
    sp(6),
    Paragraph("A Retrieval-Augmented Generation (RAG) System", SUBTITLE),
    sp(4),
    Paragraph("Technical Report &mdash; Pipeline, Techniques, and Visualisations", BYLINE),
    sp(10),
    Paragraph("Rutgers Business School &nbsp;|&nbsp; Graduate Programs", BYLINE),
    sp(6),
    hr(),
    sp(30),
]

# Quick summary box on cover
cover_table = Table([
    [Paragraph("<b>What the system does</b>", S("ct", fontSize=11,
               fontName="Helvetica-Bold", textColor=white)),
     Paragraph("Answers student questions about RBS contacts, events, academic "
               "requirements, and student organisations using real Rutgers data — "
               "with source citations and no hallucination.",
               S("cb", fontSize=10, leading=15, fontName="Helvetica",
                 textColor=white))]
], colWidths=[1.6*inch, 4.9*inch])
cover_table.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,-1), RU_RED),
    ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ("TOPPADDING",    (0,0),(-1,-1), 12),
    ("BOTTOMPADDING", (0,0),(-1,-1), 12),
    ("LEFTPADDING",   (0,0),(-1,-1), 10),
]))
story += [cover_table, sp(20)]

# Key numbers row
kn = Table([
    [Paragraph("<b>128</b><br/>chunks indexed", S("kn", fontSize=13,
               fontName="Helvetica-Bold", textColor=RU_RED, alignment=TA_CENTER)),
     Paragraph("<b>384-d</b><br/>embeddings", S("kn2", fontSize=13,
               fontName="Helvetica-Bold", textColor=BLUE, alignment=TA_CENTER)),
     Paragraph("<b>2 indexes</b><br/>FAISS + BM25", S("kn3", fontSize=13,
               fontName="Helvetica-Bold", textColor=GREEN, alignment=TA_CENTER)),
     Paragraph("<b>100 %</b><br/>hit rate (5 queries)", S("kn4", fontSize=13,
               fontName="Helvetica-Bold", textColor=ORANGE, alignment=TA_CENTER)),
     Paragraph("<b>&lt; 10 ms</b><br/>retrieval latency", S("kn5", fontSize=13,
               fontName="Helvetica-Bold", textColor=PURPLE, alignment=TA_CENTER))],
], colWidths=[1.3*inch]*5)
kn.setStyle(TableStyle([
    ("BOX",          (0,0),(-1,-1), 1, RU_GREY),
    ("INNERGRID",    (0,0),(-1,-1), 0.5, RU_GREY),
    ("BACKGROUND",   (0,0),(-1,-1), RU_LIGHT),
    ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ("TOPPADDING",   (0,0),(-1,-1), 10),
    ("BOTTOMPADDING",(0,0),(-1,-1), 10),
]))
story += [kn, PageBreak()]


# ─────────────────────── SECTION 1: WHAT IS RAG? ─────────────────────────────
story += [
    Paragraph("1. What is This Project?", H1), hr(),
    body("This project is a <b>question-answering chatbot</b> built specifically for "
         "Rutgers Business School (RBS) students. A student can type a question like "
         "<i>\"Who do I contact for the MITA program?\"</i> and the system will give "
         "a precise, cited answer drawn directly from official RBS webpages."),
    sp(4),
    body("The core design is called <b>Retrieval-Augmented Generation (RAG)</b>. "
         "The idea is simple: instead of asking an AI to recall facts from its "
         "training (which can be outdated or simply wrong), we first "
         "<i>retrieve</i> the relevant passages from our own database, then hand "
         "those passages to the AI so it can form an answer. This means every "
         "answer is grounded in real, current data — and the system can tell "
         "you exactly which webpage the answer came from."),
    sp(6),
    info_box("Why not just use ChatGPT directly?",
             "A general AI like ChatGPT does not know RBS-specific details "
             "(coordinator emails, upcoming events, credit requirements). It could "
             "guess — but guessing in this context is harmful. RAG prevents that "
             "by forcing the model to answer only from provided documents."),
    sp(8),
    body("The project has three distinct phases, each handled by a separate Python file:"),
    sp(4),
    two_col_table([
        ("ingest.py",   "Runs once offline. Reads the raw RBS text files, splits them into "
                        "manageable pieces, and builds two search indexes that are saved to disk."),
        ("retrieval.py","Runs at query time. Takes a student's question, searches both indexes, "
                        "and returns the five most relevant pieces of text."),
        ("generator.py","Takes those five pieces and asks the LLM to write a clear, cited answer "
                        "using only that information."),
        ("app.py",      "The Streamlit web interface that ties everything together into a chat UI."),
        ("evaluate.py", "Runs benchmark tests to measure how accurately the system retrieves "
                        "the right information."),
    ]),
    PageBreak(),
]


# ─────────────────────── SECTION 2: DATA PREPARATION ────────────────────────
story += [
    Paragraph("2. Data Preparation — Building the Knowledge Base", H1), hr(),
    body("Before the system can answer any questions, it needs to learn from the "
         "source material. The RBS knowledge base is stored as plain text files in "
         "the <b>data/</b> folder, one file per topic area:"),
    sp(4),
    two_col_table([
        ("contacts.txt",      "Graduate programme contact emails (MBA, MITA, MSF, etc.)"),
        ("events.txt",        "Upcoming RBS events, seminars, and open-house sessions"),
        ("requirements.txt",  "Major and minor credit requirements from the course catalogue"),
        ("student_life.txt",  "Student clubs, organisations, and the myRBS portal"),
        ("general.txt",       "General RBS homepage content and initiatives"),
        ("courses.txt",       "Course listings and descriptions"),
    ]),
    sp(10),

    Paragraph("2.1  Parsing — Reading the Raw Files", H2),
    body("Each text file uses a separator line of equals signs "
         "(<code>========</code>) to divide it into sections. "
         "The parser reads these sections and extracts metadata — the category "
         "(contacts, events, etc.), the page title, the source URL, and a short "
         "description. This metadata is attached to every chunk so that citations "
         "can be included in answers later."),
    sp(6),
    info_box("Why metadata matters",
             "Metadata prefix — e.g., \"[CONTACTS] Contact Us (https://myrbs…)\" — "
             "is prepended to every chunk before it is embedded. This means the "
             "embedding model 'sees' the category and source, which improves "
             "retrieval accuracy because the vector captures context, not just words."),
    sp(10),

    Paragraph("2.2  Chunking — Splitting Documents into Pieces", H2),
    body("Long documents are difficult to search precisely. A 2 000-word page "
         "about student clubs contains many different topics; returning the whole "
         "page in response to a question about BASE (Business Association of Supply "
         "Expertise) would drown the answer in noise."),
    sp(4),
    body("The solution is to split every document into <b>chunks of 300 words</b> "
         "with a <b>50-word overlap</b> between consecutive chunks. The overlap "
         "is critical: it ensures that a sentence that happens to sit at the "
         "boundary between two chunks is not lost — it appears in both."),
    sp(4),
    code("words = text.split()\n"
         "for i in range(0, len(words), 300 - 50):   # step = 250\n"
         "    chunk = \" \".join(words[i : i + 300])"),
    sp(4),
    body("This is <b>word-level tokenisation</b> — the text is split simply by "
         "whitespace. No NLP, no stemming, no fancy processing. It is fast and "
         "works well for the moderate-length documents in this corpus."),
    sp(10),

    Paragraph("2.3  Structured Data Extraction — Regex Patterns", H2),
    body("In addition to free-text chunks, the ingestion step extracts three "
         "types of structured facts using regular expressions (regex):"),
    sp(4),
    two_col_table([
        ("Contact emails",
         "Pattern: [\\w\\.-]+@business\\.rutgers\\.edu\n"
         "Example match: mita@business.rutgers.edu"),
        ("Event dates",
         "Pattern: [A-Z][a-z]{2}\\s\\d{1,2},\\s\\d{4}\n"
         "Example match: Mar 28, 2026"),
        ("Credit counts",
         "Pattern: (\\d+)\\s*credits\n"
         "Example match: 21 credits"),
    ]),
    sp(6),
    body("These structured extracts are saved to JSON files (contacts.json, "
         "events.json, requirements.json). They are useful for quickly "
         "verifying what factual data the system has access to."),
    PageBreak(),
]


# ─────────────────────── SECTION 3: DUAL INDEXING ───────────────────────────
story += [
    Paragraph("3. Dual Indexing — Two Ways to Search", H1), hr(),
    body("The heart of the system is its <b>hybrid retrieval</b>: two completely "
         "different search techniques are run simultaneously, and their results "
         "are combined. This is more powerful than either method alone because "
         "they are good at different things."),
    sp(8),

    Paragraph("3.1  Dense Search — Meaning-Based Retrieval (FAISS)", H2),
    body("<b>How it works:</b> Every chunk is converted into a list of 384 numbers "
         "called a <b>vector</b> or <b>embedding</b>. Similar meanings produce "
         "similar vectors — so \"contact person\" and \"who should I email\" end "
         "up close together in this 384-dimensional space even though they share "
         "no words in common."),
    sp(4),
    body("The model that creates these vectors is <b>all-MiniLM-L6-v2</b> from "
         "the SentenceTransformers library. Internally, it uses a BERT-style "
         "transformer with WordPiece subword tokenisation (up to 256 sub-tokens "
         "per input) and mean-pools the output into a single 384-d vector."),
    sp(4),
    body("At query time, the question is embedded the same way, and Facebook's "
         "<b>FAISS</b> library finds the 15 closest chunk vectors using "
         "<b>L2 (Euclidean) distance</b>. Smaller distance = more similar = "
         "more relevant. This is called an <i>IndexFlatL2</i> — it checks every "
         "single vector (no approximation), which is fine for 128 chunks."),
    sp(6),
    info_box("Dense search strength",
             "Handles paraphrases and synonyms perfectly. "
             "A query about 'how to get in touch with the programme coordinator' "
             "will still find chunks about 'email the admissions contact'."),
    sp(10),

    Paragraph("3.2  Sparse Search — Keyword-Based Retrieval (BM25)", H2),
    body("<b>How it works:</b> BM25 (Best Match 25, also called Okapi BM25) is a "
         "classical information-retrieval algorithm. It gives each chunk a score "
         "based on how often the query words appear in that chunk (<b>term "
         "frequency</b>), discounted by how common those words are across all "
         "chunks (<b>inverse document frequency</b>). Chunks that contain rare "
         "but query-relevant words score highest."),
    sp(4),
    body("Tokenisation for BM25 is deliberately simple: "
         "<b>lowercase and split on whitespace</b>. "
         "No stemming, no stop-word removal. The BM25Okapi implementation "
         "from the <i>rank_bm25</i> library is used with default "
         "parameters (k1 = 1.5, b = 0.75)."),
    sp(4),
    code("tokenized_query = query.lower().split()\n"
         "bm25_scores = bm25.get_scores(tokenized_query)"),
    sp(4),
    body("<b>Intent boost:</b> If the query router detects a specific intent "
         "(e.g., 'contacts'), every chunk from that category gets its BM25 "
         "score multiplied by <b>1.5</b> before ranking. This nudges the "
         "keyword search toward the right area of the knowledge base."),
    sp(6),
    info_box("BM25 search strength",
             "Finds exact matches that semantic search can miss. "
             "An email address like 'carmen.nieves@business.rutgers.edu' or an "
             "organisation abbreviation like 'BASE' scores very high in BM25 "
             "because those tokens appear literally in the relevant chunks."),
    PageBreak(),
]


# ─────────────────────── SECTION 4: HYBRID RETRIEVAL ────────────────────────
story += [
    Paragraph("4. Hybrid Retrieval — Combining the Two Searches", H1), hr(),
    body("Both searches return a ranked list of 15 chunks. The challenge is "
         "to merge these two lists into a single ranking. The system uses "
         "<b>Reciprocal Rank Fusion (RRF)</b>, a proven technique from "
         "information retrieval research."),
    sp(8),

    Paragraph("4.1  Reciprocal Rank Fusion (RRF)", H2),
    body("RRF works by converting each chunk's rank position into a score "
         "using the formula:"),
    sp(4),
    code("score = 1 / (60 + rank + 1)"),
    sp(4),
    body("The constant 60 prevents top-ranked chunks from receiving an "
         "overwhelmingly large score. Each chunk's scores from both lists are "
         "added together. A chunk that appears in <i>both</i> lists — ranked "
         "highly by both semantic similarity and keyword matching — will "
         "accumulate more score than one appearing in only one list. "
         "The top 5 chunks by combined RRF score are selected."),
    sp(6),
    body("<b>Example:</b> A chunk ranked #1 by FAISS gets 1/(60+0+1) = 0.0164. "
         "If the same chunk is ranked #3 by BM25, it gets an additional "
         "1/(60+2+1) = 0.0157. Total = 0.0321, which beats any chunk that "
         "only appeared in one list."),
    sp(6),
    info_box("Why RRF instead of a weighted average?",
             "Weighted averaging requires deciding how much to trust dense vs sparse "
             "— a hyperparameter that needs tuning and may change for different query "
             "types. RRF requires no tuning: it simply rewards chunks that multiple "
             "methods agree on. It is also robust to score-scale differences "
             "(BM25 scores are in the tens; cosine similarities are 0–1)."),
    sp(10),

    Paragraph("4.2  Query Router — Intent Detection", H2),
    body("Before retrieval, every query passes through a <b>keyword-based "
         "query router</b> that detects the likely topic:"),
    sp(4),
    two_col_table([
        ("contacts",     "Trigger words: contact, email, reach out, coordinator"),
        ("events",       "Trigger words: event, this week, seminar, fair"),
        ("requirements", "Trigger words: credits, minor, requirement, prerequisite"),
        ("student_life", "Trigger words: club, organization, society"),
        ("general",      "Default — no keyword matched"),
    ]),
    sp(6),
    body("The detected intent is used in two ways: (1) to apply the 1.5× BM25 "
         "boost to chunks of that category, and (2) to display the intent to the "
         "user in the spinner message so they can see what the system 'thinks' "
         "they are asking about."),
    PageBreak(),
]


# ─────────────────────── SECTION 5: GENERATION ──────────────────────────────
story += [
    Paragraph("5. Answer Generation — The Language Model", H1), hr(),
    body("Once the top 5 chunks are retrieved, they are passed to a "
         "<b>Large Language Model (LLM)</b> along with the original question. "
         "The LLM is <b>LLaMA-3.1-8b-instant</b>, accessed through the "
         "<b>Groq API</b> using the OpenAI-compatible SDK "
         "(the same Python library, just pointed at a different URL)."),
    sp(8),

    Paragraph("5.1  The Prompt Template", H2),
    body("The prompt is carefully structured to prevent hallucination:"),
    sp(4),
    code("You are the Student Life Assistant for Rutgers Business School.\n"
         "Answer ONLY from the context below.\n"
         "If the context does not contain the answer, say:\n"
         "  'I don't have information about that in my current database.'\n"
         "MUST INCLUDE CITATIONS: [Source: <url>]\n\n"
         "Context:\n"
         "--- Document 1 ---\n"
         "[CONTACTS] Contact Us (https://myrbs...)\n"
         "<chunk text here>\n"
         "...\n\n"
         "User Question: {query}"),
    sp(4),
    body("The three key design choices in the prompt:"),
    bullet("<b>Temperature = 0.0</b> — The model produces the single most "
           "likely output rather than sampling randomly. This makes answers "
           "deterministic and grounded, not creative."),
    bullet("<b>Explicit fallback instruction</b> — The model is told what to "
           "say when the context does not contain an answer. This is why the "
           "system correctly replies 'I don't have information' to "
           "'What is the capital of France?' instead of guessing Paris."),
    bullet("<b>Citation requirement</b> — Every answer must cite the source URL "
           "or file, which forces the model to trace its answer back to a "
           "specific document."),
    sp(8),

    Paragraph("5.2  The Groq / LLaMA Choice", H2),
    body("Groq provides an inference API that runs LLaMA-3.1-8b-instant on "
         "custom hardware (LPUs — Language Processing Units). This delivers "
         "response latency of ~300–600 ms, which is substantially faster than "
         "typical GPT-4 API calls. The 8-billion-parameter model is small enough "
         "to be cheap and fast, yet capable enough to follow the strict "
         "citation-only instructions reliably."),
    sp(8),

    Paragraph("5.3  Streamlit UI (app.py)", H2),
    body("The front end is built with <b>Streamlit</b>, a Python library that "
         "turns a script into an interactive web app. Key implementation choices:"),
    bullet("<b>@st.cache_resource</b> — The Retriever and RAGGenerator objects "
           "are created once when the app starts and reused for every query. "
           "This avoids reloading the 384-d FAISS index and BM25 index on "
           "every message."),
    bullet("<b>st.session_state</b> — The full conversation history is stored "
           "in Streamlit's session state so earlier messages remain visible."),
    bullet("<b>Expander for sources</b> — Each assistant response has a "
           "collapsible 'View Retrieved Sources' section so users can inspect "
           "exactly which chunks were used to generate the answer."),
    PageBreak(),
]


# ─────────────────────── SECTION 6: CHARTS ───────────────────────────────────
story += [
    Paragraph("6. Visualisation Explanations", H1), hr(),
    body("This section walks through each of the ten charts generated by "
         "visualize.py, explaining what is shown and what it tells us about "
         "the system's behaviour."),
    sp(6),
]

# ── Chart helper ──────────────────────────────────────────────────────────────
def chart_section(number, title, filename, what_it_shows, how_to_read,
                  key_insight, width=6.4*inch):
    items = [
        sp(4),
        Paragraph(f"Chart {number}: {title}", CHART_TITLE),
        thin_hr(),
        Paragraph("<b>What it shows:</b>", H3),
        body(what_it_shows),
        sp(4),
        img(filename, width=width),
        Paragraph(f"Figure {number}: {title}", CAPTION),
        sp(4),
        Paragraph("<b>How to read it:</b>", H3),
        body(how_to_read),
        sp(4),
        info_box("Key insight", key_insight, bg=HexColor("#FFF8F8")),
        sp(8),
    ]
    return items


# Chart 1
story += chart_section(
    1, "End-to-End Pipeline Architecture",
    "01_pipeline_architecture.png",
    what_it_shows=(
        "The complete flow of data through the system, split into two "
        "horizontal lanes. The <b>top lane (Offline)</b> shows everything that "
        "happens once when ingest.py is run: raw text files are parsed, "
        "chunked, embedded, and stored as FAISS and BM25 indexes. "
        "The <b>bottom lane (Online)</b> shows what happens every time a "
        "student asks a question: the query is routed, searched in parallel "
        "by FAISS and BM25, fused via RRF, and passed to the LLM."
    ),
    how_to_read=(
        "Follow the arrows. Start at 'Raw .txt files' (top-left). "
        "Each box is one processing step; the arrow shows what flows to "
        "the next step. The dashed diagonal arrow in the middle shows the "
        "pre-built indexes being loaded into memory when the app starts. "
        "The bottom lane begins at 'User Query' (bottom-left) and ends "
        "at 'Answer + Citations' (bottom-right)."
    ),
    key_insight=(
        "The offline phase runs once; the online phase runs in milliseconds. "
        "This separation is why the system can answer instantly — all the "
        "expensive embedding computation was done in advance."
    )
)

# Chart 2
story += chart_section(
    2, "Chunking Effects",
    "02_chunking_effects.png",
    what_it_shows=(
        "Two sub-plots. <b>Left:</b> a histogram showing how many words are in "
        "each chunk, with bars colour-coded by category. The red dashed line "
        "marks the 300-word target. The dotted black line marks the actual "
        "mean chunk length. <b>Right:</b> an overlap trace for one real document, "
        "showing how consecutive chunks (red bars) share a 50-word region "
        "(gold bars) at their boundaries."
    ),
    how_to_read=(
        "Left plot: most bars cluster at or below 300 words. Shorter chunks "
        "come from sections that were naturally shorter than 300 words. "
        "Right plot: each horizontal bar is one chunk. The gold region at "
        "the start of each bar is the 50 words carried over from the "
        "previous chunk. Notice the bars progressively start later in the "
        "document (shifted right) but overlap slightly."
    ),
    key_insight=(
        "The overlap is the most important detail here. Without it, an answer "
        "that spans a chunk boundary would be split across two chunks and "
        "neither chunk would contain the full context needed to answer the "
        "question correctly."
    )
)

story.append(PageBreak())

# Chart 3
story += chart_section(
    3, "Corpus Category Distribution",
    "03_category_distribution.png",
    what_it_shows=(
        "Two sub-plots. <b>Left:</b> a horizontal bar chart showing how many "
        "chunks exist per category. <b>Right:</b> a box plot showing the "
        "spread of word counts within each category, with the red dashed "
        "line at the 300-word target."
    ),
    how_to_read=(
        "Left: longer bars = more chunks = more coverage in that area. "
        "If one category has very few chunks the system will struggle to "
        "answer detailed questions about it. "
        "Right: the box shows the middle 50% of chunk lengths; "
        "the whiskers extend to the extremes. A category with boxes "
        "far below 300 words has many short documents."
    ),
    key_insight=(
        "Data imbalance between categories is a real risk: if one category "
        "has only 2–3 chunks, those chunks will dominate any query routed "
        "to that category, reducing answer diversity. This chart makes "
        "that imbalance visible at a glance."
    )
)

# Chart 4
story += chart_section(
    4, "BM25 Score Landscape",
    "04_bm25_score_landscape.png",
    what_it_shows=(
        "Six sub-plots — one for each test query — each showing the BM25 "
        "scores of the top-15 retrieved chunks. Bars are colour-coded by "
        "category (red = contacts, orange = events, blue = requirements, "
        "green = student_life, grey = general)."
    ),
    how_to_read=(
        "The longest bar is the highest-scoring chunk. For a well-formed "
        "query, the top bars should all be the same colour (the correct "
        "category). For the 'capital of France' query (out-of-scope), "
        "all bars should be very short — BM25 can't find any strong match "
        "because none of the query words appear in the knowledge base."
    ),
    key_insight=(
        "The out-of-scope query panel is the most revealing: BM25 scores "
        "collapse to near-zero, which is a natural signal that the system "
        "has nothing relevant. This is why the LLM correctly says "
        "'I don't have information' — it receives five chunks that are "
        "barely relevant at best."
    )
)

story.append(PageBreak())

# Chart 5
story += chart_section(
    5, "RRF Fusion Trace",
    "05_rrf_fusion_trace.png",
    what_it_shows=(
        "Three side-by-side bar charts for the MITA contacts query. "
        "<b>Left</b> shows the contribution of each chunk from the FAISS "
        "ranked list. <b>Middle</b> shows the contribution from the BM25 "
        "ranked list. <b>Right</b> shows the final fused RRF scores, with "
        "the top-5 cutoff marked in red."
    ),
    how_to_read=(
        "Each bar's length represents 1/(60+rank+1). A chunk that appears "
        "in both the left and middle panels will appear taller in the right "
        "panel because its contributions are summed. The red dashed line "
        "in the right panel shows the top-5 cutoff — only chunks above "
        "that line are passed to the LLM."
    ),
    key_insight=(
        "This chart shows exactly why hybrid retrieval beats either method "
        "alone. A chunk that both FAISS and BM25 agree on will accumulate "
        "more RRF score than a chunk that only one method liked. "
        "Agreement between methods = higher confidence."
    )
)

# Chart 6
story += chart_section(
    6, "t-SNE Embedding Space",
    "06_tsne_embedding_space.png",
    what_it_shows=(
        "A 2-dimensional projection of all 128 chunk embeddings. "
        "Each dot is one chunk, coloured by its category. "
        "The 384-dimensional vectors were reduced to 2D using t-SNE "
        "(t-Distributed Stochastic Neighbour Embedding). "
        "Gold stars mark the nearest chunk to each of the five test queries."
    ),
    how_to_read=(
        "Clusters of the same colour mean the embedding model places "
        "chunks from the same topic area close together in the vector "
        "space. Stars that land inside the correct colour cluster confirm "
        "that FAISS retrieval is working. An out-of-scope star that lands "
        "far from any cluster or in the wrong cluster indicates the "
        "model is doing its best with an irrelevant query."
    ),
    key_insight=(
        "t-SNE is a compression from 384 dimensions to 2, so some "
        "distortion is expected. But seeing clusters broadly align by "
        "category confirms that the embedding model has learned meaningful "
        "semantic groupings — events cluster together, contacts cluster "
        "together — without ever being told the categories explicitly."
    )
)

story.append(PageBreak())

# Chart 7
story += chart_section(
    7, "Query Router Decision Map",
    "07_query_router_map.png",
    what_it_shows=(
        "A visual decision tree of the query router. A central 'User Query' "
        "bubble at the top fans out with arrows to five intent boxes, each "
        "colour-coded. Below each intent box are the trigger keywords and "
        "three example queries that would route there. The 'general' "
        "bucket has a warning annotation because it is the catch-all for "
        "anything ambiguous or out-of-scope."
    ),
    how_to_read=(
        "Trace any query from the top bubble to its destination box. "
        "Check whether the example queries listed feel intuitively correct. "
        "Notice the 'general' bucket: any query without a keyword match "
        "ends here, which means it receives no BM25 boost — it relies "
        "entirely on the embedding model to find relevant chunks."
    ),
    key_insight=(
        "The router is a simple if/elif chain — it is fast and transparent "
        "but brittle. A query like 'Is the MITA seminar this week?' "
        "matches both 'event' keywords and 'contact' context, "
        "so the result depends on which keyword is checked first. "
        "This is a known limitation: a trained neural classifier could "
        "handle ambiguous queries more gracefully."
    )
)

# Chart 8
story += chart_section(
    8, "Dense vs Hybrid Retrieval Comparison",
    "08_retrieval_comparison.png",
    what_it_shows=(
        "Two sub-plots comparing Dense-only (FAISS) against Hybrid "
        "(FAISS + BM25 + RRF) retrieval across five real benchmark queries. "
        "<b>Left:</b> Hit@10 — did the correct chunk appear anywhere in the "
        "top-10 results? <b>Right:</b> the rank at which the first correct "
        "chunk appeared (lower rank = better; inverted axis so better is "
        "higher on the chart)."
    ),
    how_to_read=(
        "Left: a tick mark above the bar means the answer was found; a cross "
        "means it was not. Right: a point near the top of the chart means the "
        "answer was found at rank 1 or 2 (ideal). A point at rank 11 means "
        "the answer was not found in the top 10 at all. "
        "The red dotted line at rank 5 is the cutoff — only chunks ranked "
        "1–5 are passed to the LLM."
    ),
    key_insight=(
        "Wherever the green (Hybrid) line is above the blue (Dense) line, "
        "BM25 pulled the correct answer higher in the ranking. This is "
        "most visible for queries involving exact terms like email addresses "
        "or organisation names — things a keyword search handles naturally."
    )
)

story.append(PageBreak())

# Chart 9
story += chart_section(
    9, "Per-Stage Latency Breakdown",
    "09_latency_breakdown.png",
    what_it_shows=(
        "Real wall-clock timing measurements (8 repetitions, ±1 standard "
        "deviation error bars) for each stage of the retrieval pipeline: "
        "query embedding, FAISS search, BM25 scoring, and RRF fusion. "
        "LLM generation is excluded because it is an external API call "
        "whose latency depends on Groq's servers and output length. "
        "<b>Left:</b> bar chart with error bars. "
        "<b>Right:</b> pie chart showing each stage's share of total "
        "retrieval time."
    ),
    how_to_read=(
        "The bar heights are in milliseconds. Smaller is better. "
        "Error bars show variance across the 8 runs — small error bars "
        "mean consistent performance. The pie chart shows which step "
        "dominates retrieval cost. The total retrieval time is shown "
        "by the dashed red line on the bar chart."
    ),
    key_insight=(
        "Query embedding dominates retrieval time because running the "
        "transformer model is the most computationally intensive step. "
        "FAISS search and BM25 scoring take microseconds by comparison. "
        "The entire retrieval stack runs in under 10 ms on CPU — "
        "the LLM call (300–600 ms) is the dominant cost by two orders "
        "of magnitude. Optimising retrieval speed further would have "
        "negligible impact on user experience."
    )
)

# Chart 10
story += chart_section(
    10, "Failure Mode Analysis",
    "10_failure_modes.png",
    what_it_shows=(
        "Six sub-plots, each exposing a different way the system can fail "
        "or behave unexpectedly:"
        "<br/><br/>"
        "<b>(A)</b> BM25 score distribution for an in-scope vs out-of-scope query — "
        "shows the score collapse for irrelevant questions.<br/>"
        "<b>(B)</b> FAISS L2 distance curve — shows that out-of-scope queries "
        "have higher (worse) distances at every rank.<br/>"
        "<b>(C)</b> Router ambiguity matrix — six hard queries that are difficult "
        "to classify, with their actual routed intent and the plausible "
        "alternative intent shown.<br/>"
        "<b>(D)</b> BM25 boost effect — side-by-side scores before and after "
        "the 1.5× category boost for a student_life query.<br/>"
        "<b>(E)</b> Answer presence by rank — for the Finance minor query, "
        "which BM25 ranks actually contain the answer '21 credits'?<br/>"
        "<b>(F)</b> Token budget — estimated prompt length (words) as a function "
        "of how many chunks are passed to the LLM, with the model's "
        "context limit marked."
    ),
    how_to_read=(
        "<b>(A)</b>: Taller bars = higher BM25 scores = stronger match. "
        "Out-of-scope (red) bars should be much shorter than in-scope (blue). "
        "<b>(B)</b>: The out-of-scope curve (red) should be consistently above "
        "the in-scope curve (blue) — further distance = less similar. "
        "<b>(C)</b>: Red cells show where the router actually routes; the text "
        "labels on the right show the plausible alternative. "
        "<b>(D)</b>: Compare grey (before) vs green (after) bar heights — the "
        "boost lifts student_life chunks above others. "
        "<b>(E)</b>: Green bars = chunk contains the answer; red = does not. "
        "If all green bars fall after rank 5, the system will miss the answer "
        "even though it exists in the index. "
        "<b>(F)</b>: The green shaded area is the safe zone below the context "
        "limit. The star marks the current top_k = 5 setting."
    ),
    key_insight=(
        "Panel (C) is the most practically important: the keyword router "
        "can misfire when a query touches two topics at once. "
        "Panel (E) is the most subtle: the answer exists in the index but "
        "lands outside the top-5 cutoff — this is a 'retrieval miss' even "
        "though the data is present, and it will cause the LLM to say "
        "'I don't have information' incorrectly. Raising top_k helps but "
        "Panel (F) shows there is a token-budget cost to doing so."
    ),
    width=6.4 * inch
)

story.append(PageBreak())


# ─────────────────────── SECTION 7: SUMMARY TABLE ───────────────────────────
story += [
    Paragraph("7. Technology Stack at a Glance", H1), hr(),
    sp(4),
]

tech_rows = [
    ["Component", "Technology", "Why This Choice"],
    ["Embedding\nModel", "all-MiniLM-L6-v2\n(SentenceTransformers)", "Lightweight (384-d), fast on CPU, good enough for domain-specific retrieval without fine-tuning."],
    ["Vector Store", "FAISS IndexFlatL2", "Exact search with no false negatives. O(n·d) is fine for 128 chunks."],
    ["Keyword Search", "BM25Okapi\n(rank_bm25)", "Industry-standard probabilistic model. Handles exact terms (emails, acronyms) that dense search misses."],
    ["Result Fusion", "Reciprocal Rank Fusion (RRF)", "No hyperparameter tuning required. Robust to score-scale differences between dense and sparse."],
    ["LLM", "LLaMA-3.1-8b-instant\n(via Groq API)", "Fast (~300 ms), cheap, reliable instruction-following for citation-only prompts."],
    ["Tokenisation\n(BM25)", "Whitespace split\n(lowercase only)", "Fast and sufficient; BM25 does not need morphological normalisation for this corpus."],
    ["Tokenisation\n(Embedding)", "WordPiece subword\n(inside MiniLM)", "Handles out-of-vocabulary words and morphological variants automatically."],
    ["Chunking", "300 words,\n50-word overlap", "Balances precision (small chunks) against context completeness (overlap)."],
    ["Anti-Hallucination", "T=0, cite-only prompt,\nexplicit fallback phrase", "Deterministic output; model instructed to refuse rather than guess."],
    ["UI", "Streamlit", "Python-native, minimal boilerplate, built-in session state and caching."],
]

col_widths = [1.1*inch, 1.6*inch, 3.7*inch]
t = Table(tech_rows, colWidths=col_widths, repeatRows=1)
t.setStyle(TableStyle([
    # Header row
    ("BACKGROUND",   (0,0), (-1,0), RU_RED),
    ("TEXTCOLOR",    (0,0), (-1,0), white),
    ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,0), 9),
    ("ALIGN",        (0,0), (-1,0), "CENTER"),
    # Data rows
    ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
    ("FONTSIZE",     (0,1), (-1,-1), 8.5),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [HexColor("#FFFFFF"), RU_LIGHT]),
    ("VALIGN",       (0,0), (-1,-1), "TOP"),
    ("TOPPADDING",   (0,0), (-1,-1), 6),
    ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ("LEFTPADDING",  (0,0), (-1,-1), 7),
    ("BOX",          (0,0), (-1,-1), 0.8, RU_GREY),
    ("INNERGRID",    (0,0), (-1,-1), 0.4, RU_GREY),
    # First col bold
    ("FONTNAME",     (0,1), (0,-1), "Helvetica-Bold"),
    ("TEXTCOLOR",    (0,1), (0,-1), RU_RED),
]))
story += [t, sp(10)]


# ─────────────────────── SECTION 8: LIMITATIONS ──────────────────────────────
story += [
    Paragraph("8. Known Limitations and Future Improvements", H1), hr(),
    sp(4),
    two_col_table([
        ("Static data",
         "Events and contacts were scraped in March 2026. They will become "
         "stale. A production system needs periodic re-ingestion."),
        ("Keyword router brittleness",
         "The router uses a fixed keyword list. A query like 'Is the MITA "
         "seminar this week?' touches both contacts and events. A trained "
         "intent classifier (e.g., a fine-tuned BERT) would handle this better."),
        ("Retrieval miss at rank 5",
         "As shown in Chart 10(E), the correct chunk sometimes exists in "
         "the index but falls just outside the top-5 cutoff. A neural "
         "cross-encoder re-ranker (e.g., Cohere Rerank) could fix this."),
        ("No campus differentiation",
         "RBS has campuses in Newark and New Brunswick. The current system "
         "treats all chunks equally. A campus filter would improve precision."),
        ("Single-turn only",
         "The LLM does not remember previous questions in the same session. "
         "Multi-turn conversation support would require a memory mechanism."),
        ("Token budget",
         "Passing more chunks to the LLM costs more tokens. At top_k = 10 "
         "the prompt approaches the model's context limit for long chunks."),
    ], col_widths=(1.7*inch, 4.8*inch)),
    sp(12),
    PageBreak(),
]


# ─────────────────────── CLOSING ─────────────────────────────────────────────
story += [
    Paragraph("Summary", H1), hr(),
    body("This system demonstrates that a well-designed RAG pipeline can answer "
         "domain-specific questions accurately without any model fine-tuning. "
         "The key design insight is that <b>retrieval quality matters more than "
         "generation quality</b>: if the right chunks are not in the top-5, no "
         "amount of prompt engineering will produce a correct answer. "
         "That is why the system invests in two complementary retrieval methods "
         "(dense embeddings for semantic similarity, BM25 for keyword precision) "
         "and combines them with a principled fusion algorithm (RRF)."),
    sp(8),
    body("The visualisations in Section 6 are not decorative — each one was "
         "designed to expose a specific aspect of the pipeline's behaviour: "
         "how chunking creates overlap, how BM25 collapses for irrelevant queries, "
         "how RRF accumulates evidence across two ranked lists, and where the "
         "system is most likely to fail. Understanding these failure modes is "
         "the first step toward improving the system in future work."),
    sp(16),
    hr(),
    sp(4),
    Paragraph("Rutgers Business School &nbsp;|&nbsp; Student Life Assistant Project",
              BYLINE),
    Paragraph("Built with: Python &bull; SentenceTransformers &bull; FAISS &bull; "
              "BM25 &bull; Groq / LLaMA &bull; Streamlit",
              BYLINE),
]


# ── Build ──────────────────────────────────────────────────────────────────
doc.build(story)
print(f"Report saved: {OUT_FILE}  ({os.path.getsize(OUT_FILE)//1024} KB)")
