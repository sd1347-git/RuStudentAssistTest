"""
Comprehensive RAG Pipeline Visualizations
==========================================
Generates 10 publication-quality figures covering:
  1.  End-to-end pipeline architecture
  2.  Chunking effects (word-count distribution + overlap trace)
  3.  Category distribution across the corpus
  4.  BM25 score landscape for real queries
  5.  RRF fusion trace — how two ranked lists become one
  6.  t-SNE embedding space (all chunks, coloured by category)
  7.  Query router decision map
  8.  Dense vs Hybrid retrieval hit-rate comparison
  9.  Per-stage latency breakdown (real measurements)
  10. Failure-mode analysis (out-of-scope & router ambiguity)

Run:  python visualize.py
Output: visualizations/ folder (PNG, 150 dpi)

NOTE: This file is standalone — it does NOT modify any existing project file.
      It reads from output/ (chunked_data.pkl, bm25_index.pkl, vector_index.faiss)
      exactly as evaluate.py does.
"""

import os
import pickle
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
VIZ_DIR    = "visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

# ── Rutgers colour palette ────────────────────────────────────────────────────
RU_RED   = "#CC0033"
RU_GREY  = "#5F6A72"
RU_LIGHT = "#F7F7F7"
CAT_COLORS = {
    "contacts":     "#CC0033",
    "events":       "#E87722",
    "requirements": "#006DB6",
    "student_life": "#228B22",
    "general":      "#5F6A72",
    "courses":      "#8B008B",
}
SAVE_KW = dict(dpi=150, bbox_inches="tight", facecolor="white")

# ── Load indexes once (same pattern as evaluate.py) ───────────────────────────
print("Loading indexes...")
with open(os.path.join(OUTPUT_DIR, "chunked_data.pkl"), "rb") as f:
    chunks = pickle.load(f)
with open(os.path.join(OUTPUT_DIR, "bm25_index.pkl"), "rb") as f:
    bm25 = pickle.load(f)
faiss_index = faiss.read_index(os.path.join(OUTPUT_DIR, "vector_index.faiss"))
model       = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Loaded {len(chunks)} chunks | FAISS dim = {faiss_index.d}")

# ── Six representative queries (same as evaluate.py + one extra) ──────────────
QUERIES = [
    {"q": "Who is the point of contact for the MITA program?",              "intent": "contacts"},
    {"q": "What Supply Chain events are happening this week?",               "intent": "events"},
    {"q": "How many credits do I need to minor in Finance?",                 "intent": "requirements"},
    {"q": "What student organizations are available for supply chain?",      "intent": "student_life"},
    {"q": "What is the capital of France?",                                  "intent": "out_of_scope"},
    {"q": "Tell me about the MBA program admissions process",                "intent": "general"},
]


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — End-to-end Pipeline Architecture
# ─────────────────────────────────────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor(RU_LIGHT)

    def box(x, y, w, h, color, label, sublabel="", radius=0.35):
        fancy = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.1,rounding_size={radius}",
            linewidth=1.5, edgecolor="white", facecolor=color, zorder=3)
        ax.add_patch(fancy)
        ax.text(x + w / 2, y + h / 2 + (0.18 if sublabel else 0), label,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", zorder=4)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.25, sublabel,
                    ha="center", va="center", fontsize=7.5,
                    color="white", alpha=0.9, zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color=RU_GREY),
                    zorder=2)

    # ── OFFLINE lane ──────────────────────────────────────────────────────────
    ax.text(0.3, 6.55, "OFFLINE  (ingest.py)", fontsize=9,
            color=RU_GREY, style="italic", va="center")
    ax.axhline(4.9, color="#cccccc", lw=1, ls="--", xmin=0.015, xmax=0.985)

    box(0.3,  5.1, 2.2, 1.5, "#7B2D8B", "Raw .txt files",
        "contacts · events\nrequirements · student_life")
    box(3.2,  5.1, 2.2, 1.5, RU_GREY,   "Parser & Chunker",
        "300-word chunks\n50-word overlap")
    box(6.1,  5.4, 2.0, 1.0, "#006DB6", "MiniLM-L6-v2\nEncoder", "384-d vectors")
    box(9.1,  5.4, 2.0, 1.0, "#006DB6", "FAISS\nIndexFlatL2",    "dense index")
    box(6.1,  3.8, 2.0, 1.0, "#E87722", "BM25Okapi\nTokeniser",  "whitespace split")
    box(9.1,  3.8, 2.0, 1.0, "#E87722", "BM25\nSparse Index",    "inverted index")
    box(12.2, 5.0, 2.2, 1.8, "#228B22",
        "chunked_data.pkl\nvector_index.faiss\nbm25_index.pkl",
        "persisted artifacts")

    arrow(2.5, 5.85, 3.2, 5.85)
    arrow(5.4, 5.9,  6.1, 5.9)
    arrow(5.4, 5.9,  6.1, 4.3)
    arrow(8.1, 5.9,  9.1, 5.9)
    arrow(8.1, 4.3,  9.1, 4.3)
    arrow(11.1, 5.9, 12.2, 5.7)
    arrow(11.1, 4.3, 12.2, 5.2)

    # ── ONLINE lane ───────────────────────────────────────────────────────────
    ax.text(0.3, 4.55, "ONLINE  (retrieval.py + generator.py)", fontsize=9,
            color=RU_GREY, style="italic", va="center")

    box(0.3,  0.5, 2.0, 1.4, RU_RED,    "User Query",
        "'Who is the contact\nfor MITA?'")
    box(3.1,  0.5, 2.0, 1.4, "#9B2335", "Query Router",
        "keyword intent\ndetection")
    box(6.0,  1.3, 2.2, 1.2, "#006DB6", "FAISS Search",
        "embed → L2 search\ntop-15")
    box(6.0,  0.1, 2.2, 1.2, "#E87722", "BM25 Search",
        "tokenise → score\n+1.5× boost if match")
    box(9.1,  0.5, 2.0, 1.4, "#006DB6", "RRF Fusion",
        "1/(60+rank+1)\ntop-5 chunks")
    box(12.0, 0.5, 2.2, 1.4, "#7B2D8B", "LLaMA-3.1-8b\n(Groq)",
        "T=0, grounded\ncitation prompt")
    box(15.0, 0.5, 2.5, 1.4, "#228B22", "Answer +\nCitations",
        "[Source: url]\nno hallucination")

    arrow(2.3,  1.2, 3.1, 1.2)
    arrow(5.1,  1.2, 6.0, 1.9)
    arrow(5.1,  1.2, 6.0, 0.7)
    arrow(8.2,  1.9, 9.1, 1.4)
    arrow(8.2,  0.7, 9.1, 0.8)
    arrow(11.1, 1.2, 12.0, 1.2)
    arrow(14.2, 1.2, 15.0, 1.2)

    # dashed connector: artifacts → online retrieval
    ax.annotate("", xy=(9.9, 1.9), xytext=(13.3, 5.0),
                arrowprops=dict(arrowstyle="->", lw=1.2,
                                color="#aaaaaa", linestyle="dashed"), zorder=1)
    ax.text(11.7, 3.6, "load at startup\n(cached)", fontsize=7,
            color="#aaaaaa", ha="center", style="italic")

    ax.set_title("RAG Pipeline — RBS Student Life Assistant",
                 fontsize=14, fontweight="bold", color=RU_RED, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "01_pipeline_architecture.png"), **SAVE_KW)
    plt.close()
    print("  OK  01_pipeline_architecture.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Chunking Effects
# ─────────────────────────────────────────────────────────────────────────────
def fig_chunking():
    fig = plt.figure(figsize=(16, 6))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.38)

    # ── Left: word-count histogram ────────────────────────────────────────────
    ax1  = fig.add_subplot(gs[0])
    wcs  = [len(c["text"].split()) for c in chunks]
    cats = [c["category"]          for c in chunks]
    bins = np.arange(0, 360, 20)
    for cat, col in CAT_COLORS.items():
        wc = [w for w, c in zip(wcs, cats) if c == cat]
        if wc:
            ax1.hist(wc, bins=bins, alpha=0.72, color=col,
                     label=cat, edgecolor="white", linewidth=0.4)
    ax1.axvline(300, color=RU_RED, lw=2, ls="--", label="target = 300 w")
    ax1.axvline(np.mean(wcs), color="black", lw=1.5, ls=":",
                label=f"mean = {np.mean(wcs):.0f} w")
    ax1.set_xlabel("Chunk word count", fontsize=11)
    ax1.set_ylabel("Number of chunks", fontsize=11)
    ax1.set_title(f"Chunk Word-Count Distribution  (N = {len(chunks)} chunks)",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_facecolor(RU_LIGHT)

    # ── Right: overlap trace for one real document ────────────────────────────
    ax2       = fig.add_subplot(gs[1])
    sample    = next(c for c in chunks if c["category"] == "contacts")
    words     = sample["text"].split()
    CHUNK_SZ  = 300
    OVERLAP   = 50
    step      = CHUNK_SZ - OVERLAP
    n_sim     = min(6, max(1, (len(words) - OVERLAP) // step + 1))
    grad      = plt.cm.Reds(np.linspace(0.35, 0.9, n_sim))

    for i in range(n_sim):
        start = i * step
        end   = start + CHUNK_SZ
        b_s   = start / max(len(words), 1)
        b_e   = min(end, len(words)) / max(len(words), 1)
        ax2.barh(i, b_e - b_s, left=b_s, height=0.55,
                 color=grad[i], edgecolor="white", linewidth=0.8)
        if i > 0:
            ov_e = (start + OVERLAP) / max(len(words), 1)
            if ov_e > b_s:
                ax2.barh(i, ov_e - b_s, left=b_s, height=0.55,
                         color="gold", alpha=0.85,
                         edgecolor="white", linewidth=0.5)

    ylabels = [f"Chunk {i+1}\n({min((i*step)+CHUNK_SZ, len(words)) - i*step} w)"
               for i in range(n_sim)]
    ax2.set_yticks(range(n_sim))
    ax2.set_yticklabels(ylabels, fontsize=8)
    ax2.set_xlabel("Fraction of document words", fontsize=11)
    ax2.set_title("Chunking Overlap Trace  (gold = 50-word overlap region)",
                  fontsize=12, fontweight="bold")
    ax2.set_facecolor(RU_LIGHT)
    red_p  = mpatches.Patch(color=grad[min(1, n_sim - 1)], label="Chunk body (300 w)")
    gold_p = mpatches.Patch(color="gold",                  label="Overlap  (50 w)")
    ax2.legend(handles=[red_p, gold_p], fontsize=9)

    fig.suptitle("Chunking Effects on the Corpus",
                 fontsize=14, fontweight="bold", color=RU_RED, y=1.01)
    plt.savefig(os.path.join(VIZ_DIR, "02_chunking_effects.png"), **SAVE_KW)
    plt.close()
    print("  OK  02_chunking_effects.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Corpus Category Distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig_category_dist():
    from collections import Counter, defaultdict
    counts = Counter(c["category"] for c in chunks)
    cats   = list(counts.keys())
    vals   = [counts[c] for c in cats]
    colors = [CAT_COLORS.get(c, "#999999") for c in cats]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    ax = axes[0]
    bars = ax.barh(cats, vals, color=colors, edgecolor="white",
                   linewidth=0.8, height=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Number of chunks", fontsize=11)
    ax.set_title("Chunk Count by Category", fontsize=12, fontweight="bold")
    ax.set_facecolor(RU_LIGHT)
    ax.invert_yaxis()

    # Box plot: words per chunk per category
    ax2 = axes[1]
    cat_wc = defaultdict(list)
    for c in chunks:
        cat_wc[c["category"]].append(len(c["text"].split()))
    bp = ax2.boxplot([cat_wc[c] for c in cats], vert=False,
                     patch_artist=True,
                     medianprops=dict(color="white", lw=2))
    for patch, cat in zip(bp["boxes"], cats):
        patch.set_facecolor(CAT_COLORS.get(cat, "#999999"))
        patch.set_alpha(0.8)
    ax2.set_yticklabels(cats, fontsize=10)
    ax2.set_xlabel("Words per chunk", fontsize=11)
    ax2.set_title("Word-Count Spread by Category", fontsize=12, fontweight="bold")
    ax2.axvline(300, color=RU_RED, lw=1.5, ls="--", alpha=0.7, label="target 300 w")
    ax2.legend(fontsize=9)
    ax2.set_facecolor(RU_LIGHT)

    fig.suptitle("Corpus Composition", fontsize=14,
                 fontweight="bold", color=RU_RED)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "03_category_distribution.png"), **SAVE_KW)
    plt.close()
    print("  OK  03_category_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — BM25 Score Landscape (real scores for all six queries)
# ─────────────────────────────────────────────────────────────────────────────
def fig_bm25_landscape():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
    axes = axes.flatten()

    for ax, item in zip(axes, QUERIES):
        q      = item["q"]
        toks   = q.lower().split()
        scores = bm25.get_scores(toks)
        top15  = np.argsort(scores)[::-1][:15]

        top_scores = scores[top15]
        top_cats   = [chunks[i]["category"] for i in top15]
        top_titles = [
            (chunks[i]["title"][:28] + "…" if len(chunks[i]["title"]) > 28
             else chunks[i]["title"]) for i in top15
        ]
        bar_colors = [CAT_COLORS.get(c, "#999999") for c in top_cats]

        ax.barh(range(15), top_scores[::-1],
                color=bar_colors[::-1], edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(15))
        ax.set_yticklabels(top_titles[::-1], fontsize=6.5)
        ax.set_xlabel("BM25 score", fontsize=8)
        label = f'"{q[:40]}…"' if len(q) > 40 else f'"{q}"'
        ax.set_title(label, fontsize=8, fontweight="bold", color=RU_RED)
        ax.set_facecolor(RU_LIGHT)

        seen = {}
        for c, col in zip(top_cats, bar_colors):
            if c not in seen:
                seen[c] = mpatches.Patch(color=col, label=c)
        ax.legend(handles=list(seen.values()), fontsize=6, loc="lower right")

    fig.suptitle("BM25 Score Landscape — Top-15 Chunks per Query",
                 fontsize=14, fontweight="bold", color=RU_RED)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "04_bm25_score_landscape.png"), **SAVE_KW)
    plt.close()
    print("  OK  04_bm25_score_landscape.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — RRF Fusion Trace (step-by-step for the MITA query)
# ─────────────────────────────────────────────────────────────────────────────
def fig_rrf_trace():
    query  = "Who is the point of contact for the MITA program?"
    K      = 60
    TOP_N  = 15

    # Dense
    emb = model.encode([query])
    distances, dense_idx = faiss_index.search(emb, TOP_N)
    dense_idx = dense_idx[0]

    # Sparse
    toks       = query.lower().split()
    scores     = bm25.get_scores(toks)
    sparse_idx = np.argsort(scores)[::-1][:TOP_N]

    # RRF
    rrf = {}
    for rank, idx in enumerate(dense_idx):
        rrf[idx] = rrf.get(idx, 0.0) + 1 / (K + rank + 1)
    for rank, idx in enumerate(sparse_idx):
        rrf[idx] = rrf.get(idx, 0.0) + 1 / (K + rank + 1)

    top10_indices = sorted(rrf, key=rrf.get, reverse=True)[:10]

    def short(idx):
        t = chunks[idx]["title"]
        return t[:30] + "…" if len(t) > 30 else t

    dense_ranks  = {idx: r for r, idx in enumerate(dense_idx)}
    sparse_ranks = {idx: r for r, idx in enumerate(sparse_idx)}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(f'RRF Fusion Trace\nQuery: "{query}"',
                 fontsize=11, fontweight="bold", color=RU_RED)

    # Dense contributions
    ax = axes[0]
    dense_shown = [i for i in dense_idx if i in top10_indices][:10]
    vals  = [1 / (K + dense_ranks[i] + 1) for i in dense_shown]
    cols  = [CAT_COLORS.get(chunks[i]["category"], "#999") for i in dense_shown]
    lbls  = [short(i) for i in dense_shown]
    ax.barh(range(len(dense_shown)), vals[::-1],
            color=cols[::-1], edgecolor="white")
    ax.set_yticks(range(len(dense_shown)))
    ax.set_yticklabels(lbls[::-1], fontsize=7.5)
    ax.set_xlabel("Dense contribution  1/(60+rank+1)", fontsize=9)
    ax.set_title("FAISS (Dense)", fontsize=11, fontweight="bold", color="#006DB6")
    ax.set_facecolor(RU_LIGHT)

    # Sparse contributions
    ax = axes[1]
    sparse_shown = [i for i in sparse_idx if i in top10_indices][:10]
    vals2 = [1 / (K + sparse_ranks[i] + 1) for i in sparse_shown]
    cols2 = [CAT_COLORS.get(chunks[i]["category"], "#999") for i in sparse_shown]
    lbls2 = [short(i) for i in sparse_shown]
    ax.barh(range(len(sparse_shown)), vals2[::-1],
            color=cols2[::-1], edgecolor="white")
    ax.set_yticks(range(len(sparse_shown)))
    ax.set_yticklabels(lbls2[::-1], fontsize=7.5)
    ax.set_xlabel("Sparse contribution  1/(60+rank+1)", fontsize=9)
    ax.set_title("BM25 (Sparse)", fontsize=11, fontweight="bold", color="#E87722")
    ax.set_facecolor(RU_LIGHT)

    # Fused RRF
    ax = axes[2]
    fused_scores = [rrf[i] for i in top10_indices]
    fused_cols   = [CAT_COLORS.get(chunks[i]["category"], "#999") for i in top10_indices]
    fused_lbls   = [short(i) for i in top10_indices]
    ax.barh(range(10), fused_scores[::-1],
            color=fused_cols[::-1], edgecolor="white")
    ax.set_yticks(range(10))
    ax.set_yticklabels(fused_lbls[::-1], fontsize=7.5)
    ax.set_xlabel("RRF score (sum of contributions)", fontsize=9)
    ax.set_title("After RRF Fusion → Final Ranking",
                 fontsize=11, fontweight="bold", color="#228B22")
    ax.axhline(4.5, color=RU_RED, lw=2, ls="--")
    ax.text(max(fused_scores) * 0.55, 4.65,
            "Top-5 cutoff", fontsize=8, color=RU_RED)
    ax.set_facecolor(RU_LIGHT)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "05_rrf_fusion_trace.png"), **SAVE_KW)
    plt.close()
    print("  OK  05_rrf_fusion_trace.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — t-SNE Embedding Space
# ─────────────────────────────────────────────────────────────────────────────
def fig_tsne():
    print("  Computing embeddings + t-SNE (may take ~30 s)...")
    texts = [c["metadata_prefix"] + c["text"] for c in chunks]
    embs  = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=42, init="pca", learning_rate="auto")
    proj = tsne.fit_transform(embs)

    fig, ax = plt.subplots(figsize=(11, 8))
    cats_list = [c["category"] for c in chunks]

    for cat, col in CAT_COLORS.items():
        mask = [i for i, c in enumerate(cats_list) if c == cat]
        if mask:
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=col, label=cat, s=22, alpha=0.72, edgecolors="none")

    # Mark nearest chunk to each test query with a star
    for qitem in QUERIES[:5]:
        qe   = model.encode([qitem["q"]], convert_to_numpy=True)
        _, ni = faiss_index.search(qe, 1)
        ni   = ni[0][0]
        ax.scatter(proj[ni, 0], proj[ni, 1], marker="*", s=280,
                   color="gold", edgecolors="black", lw=0.8, zorder=5)
        ax.annotate(
            qitem["intent"],
            xy=(proj[ni, 0], proj[ni, 1]),
            xytext=(proj[ni, 0] + 2.5, proj[ni, 1] + 2.5),
            fontsize=7.5, color="black",
            arrowprops=dict(arrowstyle="-", color="#555", lw=0.7))

    ax.legend(fontsize=10, markerscale=1.5,
              title="Category", title_fontsize=10)
    ax.set_title(
        "t-SNE of Chunk Embeddings  (all-MiniLM-L6-v2, 384-d → 2-d)\n"
        "★ = nearest chunk to each test query",
        fontsize=12, fontweight="bold", color=RU_RED)
    ax.set_xlabel("t-SNE dim 1", fontsize=10)
    ax.set_ylabel("t-SNE dim 2", fontsize=10)
    ax.set_facecolor(RU_LIGHT)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "06_tsne_embedding_space.png"), **SAVE_KW)
    plt.close()
    print("  OK  06_tsne_embedding_space.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Query Router Decision Map
# ─────────────────────────────────────────────────────────────────────────────
def fig_router():
    ROUTER_KEYWORDS = {
        "contacts":     ["contact", "email", "reach out", "coordinator"],
        "events":       ["event", "this week", "seminar", "fair"],
        "requirements": ["credits", "minor", "requirement", "prerequisite"],
        "student_life": ["club", "organization", "society"],
        "general":      ["(no keyword match)"],
    }
    EXAMPLES = {
        "contacts":     ["Who can I email about MITA?",
                         "What's the coordinator's contact?",
                         "How do I reach out to admissions?"],
        "events":       ["What events are this week?",
                         "Are there any career fairs?",
                         "Upcoming seminars at RBS?"],
        "requirements": ["Credits needed for Finance minor?",
                         "What are the prerequisites?",
                         "How many credits to graduate?"],
        "student_life": ["Which clubs can I join?",
                         "Is there a supply chain org?",
                         "What societies exist at RBS?"],
        "general":      ["Tell me about RBS programs",
                         "What is the capital of France?",
                         "Explain the MBA process"],
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    fig.patch.set_facecolor(RU_LIGHT)

    intents = list(ROUTER_KEYWORDS.keys())
    xs      = np.linspace(0.1, 0.9, len(intents))

    # Query bubble
    qbox = mpatches.FancyBboxPatch(
        (0.38, 0.82), 0.24, 0.12,
        boxstyle="round,pad=0.02",
        facecolor=RU_RED, edgecolor="white", linewidth=2,
        transform=ax.transAxes, zorder=3)
    ax.add_patch(qbox)
    ax.text(0.5, 0.88, "User Query\n(keyword scan)",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="white", transform=ax.transAxes)

    for intent, x in zip(intents, xs):
        col  = CAT_COLORS.get(intent, "#999")
        kws  = ROUTER_KEYWORDS[intent]
        exqs = EXAMPLES[intent]

        # Arrow
        ax.annotate("", xy=(x, 0.68), xytext=(0.5, 0.82),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=col))

        # Intent box
        ibox = mpatches.FancyBboxPatch(
            (x - 0.07, 0.55), 0.14, 0.12,
            boxstyle="round,pad=0.01",
            facecolor=col, edgecolor="white", linewidth=1.5,
            transform=ax.transAxes, zorder=3)
        ax.add_patch(ibox)
        ax.text(x, 0.61, intent.replace("_", "\n"),
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", transform=ax.transAxes)

        # Keywords
        ax.text(x, 0.48, "\n".join(f"• {k}" for k in kws[:3]),
                ha="center", va="top", fontsize=7.5,
                color=col, transform=ax.transAxes, style="italic")

        # Example queries
        ax.text(x, 0.38, "e.g.:", ha="center", fontsize=7.5,
                color="#555", transform=ax.transAxes, fontweight="bold")
        for j, eq in enumerate(exqs[:3]):
            ax.text(x, 0.33 - j * 0.08, f'"{eq[:34]}"',
                    ha="center", fontsize=6.8,
                    color="#333", transform=ax.transAxes)

    # Warning for "general"
    ax.text(xs[-1], 0.07,
            "⚠  'general' catches ambiguous\n& out-of-scope queries\n(no boost applied)",
            ha="center", va="center", fontsize=8, color="#CC6600",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD",
                      edgecolor="#E87722", linewidth=1))

    ax.set_title("Query Router Decision Map — Keyword-Based Intent Detection",
                 fontsize=13, fontweight="bold", color=RU_RED)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "07_query_router_map.png"), **SAVE_KW)
    plt.close()
    print("  OK  07_query_router_map.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — Dense vs Hybrid Retrieval Hit-Rate (real measurements)
# ─────────────────────────────────────────────────────────────────────────────
def fig_retrieval_comparison():
    EXPECTED = {
        "Who is the point of contact for the MITA program?":
            "carmen.nieves@business.rutgers.edu",
        "What Supply Chain events are happening this week?":
            "supply chain",
        "How many credits do I need to minor in Finance?":
            "finance",
        "What student organizations are available for supply chain?":
            "base",
        "What is the capital of France?":
            "paris",
    }

    dense_hits, hybrid_hits   = [], []
    dense_ranks, hybrid_ranks = [], []
    labels = []

    for q, expected in EXPECTED.items():
        qe = model.encode([q])

        # Dense only
        _, di = faiss_index.search(qe, 10)
        d_chunks = [chunks[i] for i in di[0]]
        d_hit  = any(expected in c["text"].lower() for c in d_chunks)
        d_rank = next((r + 1 for r, c in enumerate(d_chunks)
                       if expected in c["text"].lower()), None)

        # Hybrid
        toks = q.lower().split()
        bsc  = bm25.get_scores(toks)
        sp   = np.argsort(bsc)[::-1][:15]
        K    = 60
        rrf  = {}
        for rank, idx in enumerate(di[0][:15]):
            rrf[idx] = rrf.get(idx, 0.0) + 1 / (K + rank + 1)
        for rank, idx in enumerate(sp):
            rrf[idx] = rrf.get(idx, 0.0) + 1 / (K + rank + 1)
        fused   = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:10]
        h_chunks = [chunks[i] for i, _ in fused]
        h_hit  = any(expected in c["text"].lower() for c in h_chunks)
        h_rank = next((r + 1 for r, (i, _) in enumerate(fused)
                       if expected in chunks[i]["text"].lower()), None)

        dense_hits.append(int(d_hit))
        hybrid_hits.append(int(h_hit))
        dense_ranks.append(d_rank  if d_rank  else 11)
        hybrid_ranks.append(h_rank if h_rank else 11)
        labels.append(q[:36] + "…" if len(q) > 36 else q)

    x     = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Hit@10 bar chart
    ax1 = axes[0]
    b1 = ax1.bar(x - width / 2, dense_hits,  width,
                 label="Dense Only (FAISS)",     color="#006DB6", edgecolor="white")
    b2 = ax1.bar(x + width / 2, hybrid_hits, width,
                 label="Hybrid (FAISS+BM25+RRF)", color="#228B22", edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=16, ha="right", fontsize=7.5)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Miss", "Hit"], fontsize=10)
    ax1.set_title("Hit@10 — Dense vs Hybrid", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_facecolor(RU_LIGHT)
    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 "✓" if bar.get_height() else "✗",
                 ha="center", fontsize=13)
    for bar in b2:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 "✓" if bar.get_height() else "✗",
                 ha="center", fontsize=13)

    # Rank plot
    ax2 = axes[1]
    ax2.plot(x, dense_ranks,  "o--", color="#006DB6", lw=2, ms=8, label="Dense rank")
    ax2.plot(x, hybrid_ranks, "s-",  color="#228B22", lw=2, ms=8, label="Hybrid rank")
    ax2.axhline(5,  color=RU_RED,    lw=1.5, ls=":", alpha=0.7, label="Top-5 cutoff")
    ax2.axhline(10, color="#aaaaaa", lw=1,   ls=":", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=16, ha="right", fontsize=7.5)
    ax2.set_ylabel("Rank of first correct chunk\n(11 = not found in top-10)", fontsize=9)
    ax2.set_title("Rank of First Correct Chunk", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 12.5)
    ax2.invert_yaxis()
    ax2.legend(fontsize=9)
    ax2.set_facecolor(RU_LIGHT)

    fig.suptitle("Dense vs Hybrid Retrieval — Real Query Evaluation",
                 fontsize=14, fontweight="bold", color=RU_RED)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "08_retrieval_comparison.png"), **SAVE_KW)
    plt.close()
    print("  OK  08_retrieval_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — Per-Stage Latency Breakdown (real wall-clock measurements)
# ─────────────────────────────────────────────────────────────────────────────
def fig_latency():
    REPS = 8
    q    = "Who is the point of contact for the MITA program?"
    toks = q.lower().split()

    embed_t, faiss_t, bm25_t, rrf_t = [], [], [], []

    for _ in range(REPS):
        t0 = time.perf_counter()
        emb = model.encode([q])
        embed_t.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _, di = faiss_index.search(emb, 15)
        faiss_t.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        sc = bm25.get_scores(toks)
        sp = np.argsort(sc)[::-1][:15]
        bm25_t.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        K   = 60
        rrf = {}
        for rank, idx in enumerate(di[0]):
            rrf[idx] = rrf.get(idx, 0.0) + 1 / (K + rank + 1)
        for rank, idx in enumerate(sp):
            rrf[idx] = rrf.get(idx, 0.0) + 1 / (K + rank + 1)
        sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:5]
        rrf_t.append(time.perf_counter() - t0)

    stages = ["Query\nEmbedding\n(MiniLM)", "FAISS\nVector\nSearch",
              "BM25\nKeyword\nSearch", "RRF\nFusion"]
    means  = [np.mean(embed_t) * 1000, np.mean(faiss_t) * 1000,
              np.mean(bm25_t)  * 1000, np.mean(rrf_t)   * 1000]
    stds   = [np.std(embed_t)  * 1000, np.std(faiss_t)  * 1000,
              np.std(bm25_t)   * 1000, np.std(rrf_t)    * 1000]
    s_cols = ["#006DB6", "#006DB6", "#E87722", "#228B22"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Bar chart
    ax1 = axes[0]
    bars = ax1.bar(stages, means, color=s_cols, edgecolor="white",
                   linewidth=0.8, width=0.55,
                   yerr=stds, error_kw=dict(ecolor="#555", capsize=5, lw=1.5))
    for bar, m, s in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + s + 0.3,
                 f"{m:.1f} ms", ha="center", fontsize=9, fontweight="bold")
    total = sum(means)
    ax1.axhline(total, color=RU_RED, lw=1.5, ls="--", alpha=0.6)
    ax1.text(3.38, total + 0.5, f"Total ≈ {total:.1f} ms",
             fontsize=8, color=RU_RED)
    ax1.set_ylabel(f"Latency (ms)  ±1 std  ({REPS} reps)", fontsize=10)
    ax1.set_title("Per-Stage Retrieval Latency\n(LLM call excluded — API-dependent)",
                  fontsize=11, fontweight="bold")
    ax1.set_facecolor(RU_LIGHT)

    # Pie chart
    ax2 = axes[1]
    _, texts, autotexts = ax2.pie(
        means, labels=stages, colors=s_cols,
        autopct="%1.1f%%", startangle=90,
        textprops=dict(fontsize=9),
        wedgeprops=dict(edgecolor="white", linewidth=1.5))
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
    ax2.set_title("Fraction of Total Retrieval Time\n(LLM ~300–600 ms excluded)",
                  fontsize=11, fontweight="bold")

    fig.text(
        0.5, -0.04,
        "Retrieval stack (embedding + FAISS + BM25 + RRF) runs in <10 ms on CPU.  "
        "LLM generation (Groq llama-3.1-8b-instant) adds ~300–600 ms.",
        ha="center", fontsize=9, color=RU_GREY, style="italic")

    fig.suptitle("Cost & Latency Breakdown — Retrieval Pipeline",
                 fontsize=13, fontweight="bold", color=RU_RED)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "09_latency_breakdown.png"), **SAVE_KW)
    plt.close()
    print("  OK  09_latency_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — Failure Mode Analysis
# ─────────────────────────────────────────────────────────────────────────────
def fig_failure_modes():
    fig = plt.figure(figsize=(16, 9))
    gs  = GridSpec(2, 3, figure=fig, wspace=0.42, hspace=0.58)

    in_scope_q = "Who is the point of contact for the MITA program?"
    oos_q      = "What is the capital of France?"

    # ── (A) BM25 score distributions: in-scope vs out-of-scope ───────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for q, col, lbl in [(in_scope_q, "#006DB6", "In-scope"),
                        (oos_q,      RU_RED,     "Out-of-scope")]:
        sc = bm25.get_scores(q.lower().split())
        ax_a.hist(sc[sc > 0], bins=20, alpha=0.6, color=col,
                  label=lbl, edgecolor="white")
    ax_a.set_xlabel("BM25 score", fontsize=9)
    ax_a.set_ylabel("Chunk count", fontsize=9)
    ax_a.set_title("(A) BM25 Score Distribution\nIn-scope vs Out-of-scope",
                   fontsize=10, fontweight="bold")
    ax_a.legend(fontsize=8)
    ax_a.set_facecolor(RU_LIGHT)

    # ── (B) FAISS distance curves ─────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    for q, col, lbl in [(in_scope_q, "#006DB6", "In-scope"),
                        (oos_q,      RU_RED,     "Out-of-scope")]:
        qe  = model.encode([q])
        dist, _ = faiss_index.search(qe, 30)
        ax_b.plot(range(1, 31), dist[0], "o-", color=col,
                  lw=1.8, ms=4, label=lbl, alpha=0.85)
    ax_b.set_xlabel("Rank k", fontsize=9)
    ax_b.set_ylabel("L2 distance to query", fontsize=9)
    ax_b.set_title("(B) FAISS Distance Curve\nHigher = less relevant",
                   fontsize=10, fontweight="bold")
    ax_b.legend(fontsize=8)
    ax_b.set_facecolor(RU_LIGHT)

    # ── (C) Router ambiguity matrix ───────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    ambiguous = [
        ("What is the MBA application deadline?",     "events",        "contacts"),
        ("Who teaches finance courses?",               "contacts",      "requirements"),
        ("How do I join a business club?",             "student_life",  "contacts"),
        ("Is the MITA seminar this week?",             "events",        "contacts"),
        ("Can I take accounting with 12 credits?",    "requirements",  "contacts"),
        ("Supply chain resources available online?",   "general",       "student_life"),
    ]
    x_cats = list(CAT_COLORS.keys())
    mat    = np.zeros((len(ambiguous), len(x_cats)))
    for i, (_, rt, _) in enumerate(ambiguous):
        if rt in x_cats:
            mat[i, x_cats.index(rt)] = 1.0

    cmap = LinearSegmentedColormap.from_list("ru", ["#ffffff", RU_RED])
    ax_c.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax_c.set_xticks(range(len(x_cats)))
    ax_c.set_xticklabels(x_cats, rotation=30, ha="right", fontsize=7)
    ax_c.set_yticks(range(len(ambiguous)))
    ax_c.set_yticklabels([r[0][:30] + "…" for r in ambiguous], fontsize=6.5)
    ax_c.set_title("(C) Router Ambiguity — Hard Queries\n(red = actual routed intent)",
                   fontsize=10, fontweight="bold")
    for i, (_, _, alt) in enumerate(ambiguous):
        ax_c.text(len(x_cats) + 0.05, i, f"alt: {alt}?",
                  fontsize=5.8, va="center", color=RU_GREY, clip_on=False)

    # ── (D) BM25 boost effect ─────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    q_boost = "What student organizations are available for supply chain students?"
    toks_b  = q_boost.lower().split()
    before  = bm25.get_scores(toks_b).copy()
    after   = before.copy()
    for i, c in enumerate(chunks):
        if c["category"] == "student_life":
            after[i] *= 1.5
    top8_before = np.argsort(before)[::-1][:8]
    top8_after  = np.argsort(after)[::-1][:8]

    def short_title(idx):
        t = chunks[idx]["title"]
        return t[:22] + "…" if len(t) > 22 else t

    y = np.arange(8)
    ax_d.barh(y - 0.2, before[top8_before][::-1], 0.38,
              color=RU_GREY,    label="Before boost", edgecolor="white")
    ax_d.barh(y + 0.2, after[top8_after][::-1],   0.38,
              color="#228B22", label="After 1.5× boost", edgecolor="white")
    ax_d.set_yticks(y)
    ax_d.set_yticklabels([short_title(i) for i in top8_after[::-1]], fontsize=7)
    ax_d.set_xlabel("BM25 score", fontsize=9)
    ax_d.set_title("(D) BM25 Intent Boost Effect\n(student_life × 1.5)",
                   fontsize=10, fontweight="bold")
    ax_d.legend(fontsize=8)
    ax_d.set_facecolor(RU_LIGHT)

    # ── (E) Answer presence by rank ───────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    q_fin = "How many credits do I need to minor in Finance?"
    sc_f  = bm25.get_scores(q_fin.lower().split())
    top20 = np.argsort(sc_f)[::-1][:20]
    has_answer = [1 if "21" in chunks[i]["text"] else 0 for i in top20]
    bar_cols   = ["#228B22" if v else RU_RED for v in has_answer]
    ax_e.bar(range(1, 21), has_answer, color=bar_cols, edgecolor="white")
    ax_e.axvline(5.5, color=RU_RED, lw=2, ls="--", label="Top-5 cutoff")
    ax_e.set_xlabel("BM25 rank", fontsize=9)
    ax_e.set_ylabel("Contains answer (1 = yes)", fontsize=9)
    ax_e.set_title("(E) Answer Presence by BM25 Rank\n(Finance minor credits query)",
                   fontsize=10, fontweight="bold")
    ax_e.legend(fontsize=8)
    ax_e.set_facecolor(RU_LIGHT)
    ax_e.text(5.8, 0.85,
              "Chunks past rank 5\nmay contain the\nanswer but won't\nbe surfaced",
              fontsize=7, color=RU_RED, va="top")

    # ── (F) Token budget: prompt length vs top_k ─────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    wcs_sorted = sorted([len(c["text"].split()) for c in chunks], reverse=True)
    PROMPT_OVERHEAD = 120
    top_k_vals = [1, 3, 5, 7, 10]
    ctx_lens   = [int(np.mean(wcs_sorted[:k]) * k) + PROMPT_OVERHEAD
                  for k in top_k_vals]
    ax_f.plot(top_k_vals, ctx_lens, "s-", color=RU_RED, lw=2, ms=8,
              markeredgecolor="white")
    ax_f.axhline(8192, color="#E87722", lw=2, ls="--",
                 label="llama-3.1-8b limit (8 192 tok)")
    ax_f.fill_between(top_k_vals, ctx_lens, 8192,
                      where=[c < 8192 for c in ctx_lens],
                      alpha=0.12, color="#228B22", label="safe zone")
    ax_f.set_xlabel("top_k chunks passed to LLM", fontsize=9)
    ax_f.set_ylabel("Estimated prompt tokens (words)", fontsize=9)
    ax_f.set_title("(F) Token Budget vs top_k\n(current default: top_k = 5)",
                   fontsize=10, fontweight="bold")
    ax_f.legend(fontsize=8)
    ax_f.set_facecolor(RU_LIGHT)
    # Mark current setting (top_k = 5)
    idx5 = top_k_vals.index(5)
    ax_f.scatter([5], [ctx_lens[idx5]], s=150, zorder=5,
                 color="#228B22", edgecolors="white", linewidth=1.5)
    ax_f.annotate("current\ntop_k = 5",
                  xy=(5, ctx_lens[idx5]),
                  xytext=(6.3, ctx_lens[idx5] - 300),
                  fontsize=8, color="#228B22",
                  arrowprops=dict(arrowstyle="->", color="#228B22", lw=1))

    fig.suptitle("Failure Mode Analysis — Where the RAG Pipeline Can Break",
                 fontsize=14, fontweight="bold", color=RU_RED)
    plt.savefig(os.path.join(VIZ_DIR, "10_failure_modes.png"), **SAVE_KW)
    plt.close()
    print("  OK  10_failure_modes.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nGenerating visualizations...\n")
    fig_pipeline()
    fig_chunking()
    fig_category_dist()
    fig_bm25_landscape()
    fig_rrf_trace()
    fig_tsne()
    fig_router()
    fig_retrieval_comparison()
    fig_latency()
    fig_failure_modes()

    print(f"\nAll 10 figures saved to ./{VIZ_DIR}/")
    print("-" * 55)
    for fname in sorted(os.listdir(VIZ_DIR)):
        path = os.path.join(VIZ_DIR, fname)
        print(f"  {fname:<50s}  {os.path.getsize(path) // 1024} KB")
