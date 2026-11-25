import html
import os
import random
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# -------------------- CONFIGURATION --------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "lfw_funneled"
MODEL_PATH = BASE_DIR / "face_recognition_pca_model.pkl"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

st.set_page_config(
    page_title="EigenStudio · LFW Face EDA & PCA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- CUSTOM THEME & CSS --------------------

STUDIO_CSS = """
<style>
    [data-testid="stMainBlockContainer"],
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 2.5rem;
        max-width: 1260px;
    }
    .hero-card {
        box-sizing: border-box;
        width: 100%;
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 55%, #0C4A6E 100%);
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-radius: 18px;
        padding: 1.5rem 1.85rem;
        margin-top: 0.25rem;
        margin-bottom: 1.15rem;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
    }
    .hero-badge {
        display: inline-block;
        padding: 0.24rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.14);
        border: 1px solid rgba(56, 189, 248, 0.32);
        border-radius: 999px;
        margin-bottom: 0.65rem;
    }
    .hero-title {
        font-size: clamp(1.5rem, 2.5vw, 2rem);
        font-weight: 750;
        line-height: 1.2;
        color: #F8FAFC;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        color: #CBD5E1;
        font-size: 0.96rem;
        line-height: 1.45;
        margin: 0;
    }
    .kpi-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .kpi-card:hover {
        border-color: rgba(56, 189, 248, 0.45);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.15;
    }
    .kpi-delta {
        font-size: 0.8rem;
        color: #38BDF8;
        margin-top: 0.25rem;
    }
    .section-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
    }
    .verdict-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14), rgba(56, 189, 248, 0.1));
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 16px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 0.9rem;
    }
    .verdict-box.mismatch {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.14), rgba(251, 146, 60, 0.08));
        border-color: rgba(244, 63, 94, 0.4);
    }
    .verdict-kicker {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 0.2rem;
    }
    .verdict-name {
        font-size: 1.55rem;
        font-weight: 750;
        color: #F8FAFC;
        margin: 0;
    }
    .verdict-meta {
        margin-top: 0.35rem;
        font-size: 0.88rem;
        color: #CBD5E1;
    }
    .pipeline-pill {
        display: inline-block;
        background: #1E293B;
        border: 1px solid #334155;
        color: #E2E8F0;
        border-radius: 8px;
        padding: 0.28rem 0.6rem;
        font-size: 0.8rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .score-row {
        margin-bottom: 0.55rem;
    }
    .score-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        color: #E2E8F0;
        margin-bottom: 0.2rem;
    }
    .score-track {
        width: 100%;
        height: 8px;
        background: #1E293B;
        border-radius: 999px;
        overflow: hidden;
    }
    .score-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
    }
</style>
"""


def pretty_name(raw: str) -> str:
    """Convert folder label like George_W_Bush into clean display name."""
    return raw.replace("_", " ")


def render_kpi_card(label: str, value: str, delta: str = "") -> str:
    """Return HTML markup for a styled KPI card."""
    delta_html = f'<div class="kpi-delta">{html.escape(delta)}</div>' if delta else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f"{delta_html}"
        f"</div>"
    )


def style_dark_plot(fig: plt.Figure, ax: plt.Axes) -> None:
    """Apply cohesive dark studio styling to a Matplotlib figure and axes."""
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#0F172A")
    ax.tick_params(colors="#CBD5E1", labelsize=9)
    ax.xaxis.label.set_color("#CBD5E1")
    ax.yaxis.label.set_color("#CBD5E1")
    ax.title.set_color("#F8FAFC")
    for spine in ax.spines.values():
        spine.set_color("#1E293B")


# -------------------- CACHED DATA & MODEL LOADERS --------------------


@st.cache_data(show_spinner="Indexing lfw_funneled dataset...")
def index_dataset(dataset_path: str) -> pd.DataFrame:
    """Fast directory scan using os.scandir to index all people and images."""
    root = Path(dataset_path)
    if not root.exists():
        return pd.DataFrame(columns=["person", "image_count", "image_paths"])

    records = []
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue
            with os.scandir(entry.path) as files:
                imgs = sorted(
                    f.path
                    for f in files
                    if f.is_file() and f.name.lower().endswith(IMAGE_EXTENSIONS)
                )
            if imgs:
                records.append(
                    {
                        "person": entry.name,
                        "image_count": len(imgs),
                        "image_paths": imgs,
                    }
                )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("image_count", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def inspect_image_properties(sample_paths: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Inspect image dimensions, color mode, and pixel intensity histogram."""
    rows = []
    hist_accum = np.zeros(32, dtype=np.int64)
    for path in sample_paths:
        with Image.open(path) as img:
            arr = np.asarray(img, dtype=np.float32)
            rows.append(
                {
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "mean_brightness": float(arr.mean()),
                    "std_brightness": float(arr.std()),
                }
            )
            gray_arr = np.asarray(img.convert("L"), dtype=np.uint8)
            counts, _ = np.histogram(gray_arr, bins=32, range=(0, 256))
            hist_accum += counts
    return pd.DataFrame(rows), hist_accum


@st.cache_data(show_spinner="Loading grayscale face matrix...")
def load_filtered_faces(
    df_filtered_records: list[tuple[str, list[str]]],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load, convert to grayscale, resize, and flatten faces for PCA."""
    X_list, y_list, paths_list = [], [], []
    for person, paths in df_filtered_records:
        for path in paths:
            with Image.open(path) as img:
                gray = img.convert("L").resize((width, height))
                arr = np.asarray(gray, dtype=np.float32) / 255.0
                X_list.append(arr.flatten())
                y_list.append(person)
                paths_list.append(path)
    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list), paths_list


@st.cache_resource(show_spinner="Training PCA + SVM pipeline...")
def fit_pca_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    test_size: float = 0.25,
    svm_c: float = 10.0,
) -> dict:
    """Fit randomized PCA + RBF SVM pipeline and evaluate on a stratified holdout split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    max_components = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(
        n_components=max_components,
        svd_solver="randomized",
        whiten=True,
        random_state=42,
    )
    clf = SVC(C=svm_c, kernel="rbf", random_state=42)
    pipeline = Pipeline([("pca", pca), ("classifier", clf)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {
        "pipeline": pipeline,
        "pca": pca,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": float(acc),
        "n_components_used": int(max_components),
    }


@st.cache_resource(show_spinner="Loading saved PCA model...")
def load_pretrained_model(model_path: str) -> dict | None:
    """Load pre-trained joblib PCA model if available on disk."""
    path = Path(model_path)
    return joblib.load(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def compute_pca_gallery_index(
    _pca: PCA,
    class_names: tuple[str, ...],
    person_to_paths: dict[str, list[str]],
    width: int,
    height: int,
    max_per_person: int = 8,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Project a subset of dataset faces into PCA space for nearest-neighbor visual matching."""
    vectors, labels, paths = [], [], []
    for person in class_names:
        for path in person_to_paths.get(person, [])[:max_per_person]:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("L").resize((width, height)), dtype=np.float32) / 255.0
                vectors.append(arr.flatten())
                labels.append(person)
                paths.append(path)
    X_sub = np.asarray(vectors, dtype=np.float32)
    Z_sub = _pca.transform(X_sub)
    return Z_sub, labels, paths


# -------------------- VISUALIZATION HELPERS --------------------


def reconstruct_with_k(pca: PCA, x_flat: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Reconstruct a flattened face using the first k principal components."""
    k = max(1, min(k, pca.components_.shape[0]))
    centered = x_flat - pca.mean_
    comps_k = pca.components_[:k]
    coeffs_k = centered @ comps_k.T
    x_recon = pca.mean_ + coeffs_k @ comps_k
    mse = float(np.mean((x_flat - x_recon) ** 2))
    return x_recon, mse


def rank_candidates(pipeline: Pipeline, x_flat: np.ndarray, top_k: int = 5) -> list[dict]:
    """Return top-k candidate classes with relative match scores normalized via softmax."""
    classes = pipeline.classes_
    pred = pipeline.predict(x_flat)[0]

    if hasattr(pipeline, "decision_function"):
        scores = np.asarray(pipeline.decision_function(x_flat)).ravel()
        if scores.shape[0] == len(classes):
            shifted = scores - np.max(scores)
            probs = np.exp(shifted) / np.sum(np.exp(shifted))
            order = np.argsort(scores)[::-1][:top_k]
            return [
                {
                    "person": classes[i],
                    "score": float(scores[i]),
                    "pct": float(probs[i] * 100),
                }
                for i in order
            ]

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(x_flat)[0]
        order = np.argsort(probs)[::-1][:top_k]
        return [
            {"person": classes[i], "score": float(probs[i]), "pct": float(probs[i] * 100)}
            for i in order
        ]

    return [{"person": pred, "score": 1.0, "pct": 100.0}]


def plot_eigenfaces_grid(
    pca: PCA, height: int, width: int, n_show: int = 12, cols: int = 4
) -> plt.Figure:
    """Plot top eigenfaces with explained variance percentages on dark canvas."""
    n_show = min(n_show, pca.components_.shape[0])
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.55))
    fig.patch.set_facecolor("#111827")
    axes = np.atleast_2d(axes).ravel()

    for idx, ax in enumerate(axes):
        ax.set_facecolor("#111827")
        if idx < n_show:
            eigenface = pca.components_[idx].reshape((height, width))
            var_pct = pca.explained_variance_ratio_[idx] * 100
            ax.imshow(eigenface, cmap="bone")
            ax.set_title(f"PC {idx + 1} · {var_pct:.1f}%", fontsize=9, color="#E2E8F0")
        ax.axis("off")

    fig.tight_layout(pad=0.8)
    return fig


# -------------------- VIEW 1: DATASET STUDIO (EDA) --------------------


def render_eda_view(df_people: pd.DataFrame, df_filtered: pd.DataFrame, min_faces: int) -> None:
    total_people = len(df_people)
    total_images = int(df_people["image_count"].sum())
    single_people = int((df_people["image_count"] == 1).sum())
    filtered_people = len(df_filtered)
    filtered_images = int(df_filtered["image_count"].sum()) if not df_filtered.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(render_kpi_card("Total Identities", f"{total_people:,}", "Distinct folders"), unsafe_allow_html=True)
    k2.markdown(render_kpi_card("Total Portraits", f"{total_images:,}", f"Avg {total_images / total_people:.2f} / person"), unsafe_allow_html=True)
    k3.markdown(
        render_kpi_card(
            "Single-Photo Identities",
            f"{single_people:,}",
            f"{single_people / total_people * 100:.1f}% long tail",
        ),
        unsafe_allow_html=True,
    )
    k4.markdown(
        render_kpi_card(
            f"PCA Cohort (≥ {min_faces})",
            f"{filtered_people:,} people",
            f"{filtered_images:,} images ({filtered_images / total_images * 100:.1f}%)",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 0.8rem'></div>", unsafe_allow_html=True)
    c_left, c_right = st.columns([1.05, 1])

    with c_left:
        st.markdown("#### Most Represented Identities")
        top_n = st.slider("Show top people", min_value=5, max_value=25, value=12, key="eda_top_n")
        top_df = df_people.head(top_n).sort_values("image_count", ascending=True)
        fig_top, ax_top = plt.subplots(figsize=(6.2, max(3.6, top_n * 0.3)))
        style_dark_plot(fig_top, ax_top)
        labels = [pretty_name(p) for p in top_df["person"]]
        bars = ax_top.barh(labels, top_df["image_count"], color="#38BDF8", alpha=0.9)
        for bar in bars:
            w = bar.get_width()
            ax_top.text(w + 6, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", color="#E2E8F0", fontsize=8.5)
        ax_top.set_xlim(0, top_df["image_count"].max() * 1.14)
        ax_top.set_xlabel("Images")
        ax_top.grid(axis="x", color="#1E293B", linestyle="--", alpha=0.6)
        fig_top.tight_layout()
        st.pyplot(fig_top)
        plt.close(fig_top)

    with c_right:
        st.markdown("#### Class Imbalance Distribution")
        st.caption("Most identities appear once, while a small cohort has dozens of portraits suitable for PCA.")
        fig_dist, ax_dist = plt.subplots(figsize=(6.2, 4.15))
        style_dark_plot(fig_dist, ax_dist)
        ax_dist.hist(df_people["image_count"], bins=40, color="#818CF8", edgecolor="#0F172A", log=True)
        ax_dist.axvline(min_faces, color="#38BDF8", linestyle="--", linewidth=1.8, label=f"PCA threshold = {min_faces}")
        ax_dist.set_xlabel("Images per Identity")
        ax_dist.set_ylabel("People Count (log scale)")
        ax_dist.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")
        ax_dist.grid(alpha=0.25, color="#334155")
        fig_dist.tight_layout()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

    st.markdown("---")
    st.markdown("#### Image Anatomy & Brightness Profile")
    sample_pool = [p for paths in df_people["image_paths"] for p in paths[:2]]
    sample_paths = random.Random(42).sample(sample_pool, min(120, len(sample_pool)))
    df_props, hist_accum = inspect_image_properties(sample_paths)

    p_left, p_right = st.columns([1, 1.35])
    with p_left:
        s1, s2 = st.columns(2)
        s1.markdown(render_kpi_card("Resolution", f"{int(df_props['width'].mode()[0])}×{int(df_props['height'].mode()[0])}", "Uniform across sample"), unsafe_allow_html=True)
        s2.markdown(render_kpi_card("Color Channels", str(df_props["mode"].mode()[0]), "Converted to L for PCA"), unsafe_allow_html=True)
        st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)
        s3, s4 = st.columns(2)
        s3.markdown(render_kpi_card("Mean Brightness", f"{df_props['mean_brightness'].mean():.1f}", "0–255 luminance"), unsafe_allow_html=True)
        s4.markdown(render_kpi_card("Mean Contrast (SD)", f"{df_props['std_brightness'].mean():.1f}", "Pixel spread"), unsafe_allow_html=True)

    with p_right:
        fig_hist, ax_hist = plt.subplots(figsize=(6.5, 2.75))
        style_dark_plot(fig_hist, ax_hist)
        centers = np.linspace(4, 252, 32)
        ax_hist.fill_between(centers, hist_accum / hist_accum.sum() * 100, color="#38BDF8", alpha=0.35)
        ax_hist.plot(centers, hist_accum / hist_accum.sum() * 100, color="#38BDF8", linewidth=2)
        ax_hist.set_xlabel("Grayscale Pixel Intensity (0 = black, 255 = white)")
        ax_hist.set_ylabel("Pixel Share (%)")
        ax_hist.set_title("Aggregate Luminance Curve (120 Sampled Faces)")
        ax_hist.grid(alpha=0.2, color="#334155")
        fig_hist.tight_layout()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    st.markdown("---")
    st.markdown("#### Identity Contact Sheet & Personal Average Face")
    gallery_source = df_filtered if not df_filtered.empty else df_people
    quick_picks = gallery_source["person"].head(5).tolist()
    if "eda_selected_person" not in st.session_state or st.session_state["eda_selected_person"] not in gallery_source["person"].values:
        st.session_state["eda_selected_person"] = quick_picks[0]

    chip_cols = st.columns(len(quick_picks))
    for i, person_key in enumerate(quick_picks):
        if chip_cols[i].button(pretty_name(person_key), key=f"chip_{person_key}", width="stretch"):
            st.session_state["eda_selected_person"] = person_key

    selected_person = st.selectbox(
        "Or search any identity in the active cohort",
        gallery_source["person"].tolist(),
        index=gallery_source["person"].tolist().index(st.session_state["eda_selected_person"]),
        format_func=pretty_name,
    )
    st.session_state["eda_selected_person"] = selected_person

    person_paths = gallery_source.loc[gallery_source["person"] == selected_person, "image_paths"].iloc[0]
    g_cols = st.columns([1.15, 3.85])
    with g_cols[0]:
        stack = [np.asarray(Image.open(p).convert("L").resize((120, 120)), dtype=np.float32) / 255.0 for p in person_paths[:25]]
        avg_face = np.mean(stack, axis=0)
        st.image(avg_face, caption=f"Average Face ({len(stack)} photos)", width="stretch")
    with g_cols[1]:
        img_cols = st.columns(5)
        for idx, p in enumerate(person_paths[:10]):
            with img_cols[idx % 5]:
                st.image(Image.open(p), caption=f"#{idx + 1}", width="stretch")


# -------------------- VIEW 2: EIGENFACE LAB (PCA) --------------------


def render_pca_view(df_filtered: pd.DataFrame, width: int, height: int, n_components: int) -> None:
    if df_filtered.shape[0] < 2:
        st.warning("Lower the minimum images threshold in PCA Settings so at least 2 people are included.")
        return

    records = list(zip(df_filtered["person"], df_filtered["image_paths"]))
    X, y, img_paths = load_filtered_faces(records, width, height)
    results = fit_pca_pipeline(X, y, n_components)
    pca: PCA = results["pca"]
    k_used = results["n_components_used"]
    compression = (1 - k_used / X.shape[1]) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(render_kpi_card("Raw Pixel Space", f"{X.shape[1]:,} dims", f"{width}×{height} grayscale"), unsafe_allow_html=True)
    m2.markdown(render_kpi_card("PCA Subspace", f"{k_used} dims", f"{compression:.1f}% size reduction"), unsafe_allow_html=True)
    m3.markdown(render_kpi_card("Variance Retained", f"{pca.explained_variance_ratio_.sum() * 100:.1f}%", "Across top eigenfaces"), unsafe_allow_html=True)
    m4.markdown(render_kpi_card("Holdout SVM Accuracy", f"{results['accuracy'] * 100:.1f}%", f"{len(df_filtered)} people · 25% test"), unsafe_allow_html=True)

    st.markdown("<div style='height: 0.8rem'></div>", unsafe_allow_html=True)
    c_mean, c_var = st.columns([1, 2.2])
    with c_mean:
        st.markdown("#### Dataset Mean Face")
        mean_face = np.clip(pca.mean_.reshape((height, width)), 0, 1)
        st.image(mean_face, caption="Baseline average subtracted before PCA", width=220)

    with c_var:
        st.markdown("#### Cumulative Explained Variance")
        cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
        fig_var, ax_var = plt.subplots(figsize=(7, 3.0))
        style_dark_plot(fig_var, ax_var)
        xs = np.arange(1, len(cum_var) + 1)
        ax_var.fill_between(xs, cum_var, color="#818CF8", alpha=0.25)
        ax_var.plot(xs, cum_var, color="#38BDF8", linewidth=2.2)
        ax_var.axhline(90, color="#10B981", linestyle="--", linewidth=1.2, label="90% variance target")
        ax_var.set_xlabel("Number of Principal Components")
        ax_var.set_ylabel("Variance (%)")
        ax_var.set_ylim(0, 102)
        ax_var.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")
        ax_var.grid(alpha=0.22, color="#334155")
        fig_var.tight_layout()
        st.pyplot(fig_var)
        plt.close(fig_var)

    st.markdown("---")
    st.markdown("#### Top Eigenfaces (Principal Components)")
    n_eigen = st.slider("Eigenfaces to inspect", min_value=4, max_value=24, value=12, step=4)
    fig_eig = plot_eigenfaces_grid(pca, height, width, n_show=n_eigen)
    st.pyplot(fig_eig)
    plt.close(fig_eig)

    st.markdown("---")
    st.markdown("#### Build-a-Face: Interactive PCA Reconstruction")
    col_pick1, col_pick2, col_pick3 = st.columns([1.4, 1.4, 0.8])
    with col_pick1:
        rec_person = st.selectbox("Identity", df_filtered["person"].tolist(), format_func=pretty_name, key="lab_person")
    person_indices = [i for i, label in enumerate(y) if label == rec_person]
    with col_pick2:
        img_idx = st.selectbox(
            "Portrait",
            person_indices,
            format_func=lambda idx: Path(img_paths[idx]).name,
            key="lab_img",
        )
    with col_pick3:
        k_recon = st.slider("Custom k", min_value=1, max_value=k_used, value=min(60, k_used), key="lab_k")

    x_orig = X[img_idx : img_idx + 1]
    x_custom, mse_custom = reconstruct_with_k(pca, x_orig, k_recon)

    stage_ks = sorted({k for k in [5, 25, 75, k_used] if k <= k_used})
    st.caption("Progressive reconstruction across component counts:")
    prog_cols = st.columns(len(stage_ks) + 2)
    with prog_cols[0]:
        st.image(np.clip(x_orig.reshape((height, width)), 0, 1), caption="Original", width="stretch")
    for col_idx, sk in enumerate(stage_ks, start=1):
        xk, msek = reconstruct_with_k(pca, x_orig, sk)
        with prog_cols[col_idx]:
            st.image(np.clip(xk.reshape((height, width)), 0, 1), caption=f"k = {sk} (MSE {msek:.3f})", width="stretch")
    with prog_cols[-1]:
        diff = np.abs(x_orig - x_custom).reshape((height, width))
        fig_d, ax_d = plt.subplots(figsize=(2.4, 2.4))
        fig_d.patch.set_facecolor("#111827")
        ax_d.imshow(diff, cmap="inferno")
        ax_d.set_title(f"|Error| k={k_recon}\nMSE {mse_custom:.4f}", color="#E2E8F0", fontsize=8.5)
        ax_d.axis("off")
        fig_d.tight_layout(pad=0.2)
        st.pyplot(fig_d)
        plt.close(fig_d)

    st.markdown("---")
    st.markdown("#### 2D Face Space Map (Top 5 Identities)")
    top5 = df_filtered["person"].head(5).tolist()
    mask = np.isin(y, top5)
    Z_sub = pca.transform(X[mask])
    y_sub = y[mask]
    palette = ["#38BDF8", "#818CF8", "#34D399", "#F472B6", "#FBBF24"]

    fig_sc, ax_sc = plt.subplots(figsize=(7.5, 4.2))
    style_dark_plot(fig_sc, ax_sc)
    for idx, person in enumerate(top5):
        pts = Z_sub[y_sub == person]
        ax_sc.scatter(pts[:, 0], pts[:, 1], label=pretty_name(person), color=palette[idx % len(palette)], alpha=0.72, s=34)
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        ax_sc.scatter([cx], [cy], color=palette[idx % len(palette)], edgecolor="#FFFFFF", s=140, marker="D", zorder=5)
    ax_sc.set_xlabel("Principal Component 1")
    ax_sc.set_ylabel("Principal Component 2")
    ax_sc.set_title("Diamonds mark each person's centroid in PCA space")
    ax_sc.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8.5)
    ax_sc.grid(alpha=0.2, color="#334155")
    fig_sc.tight_layout()
    st.pyplot(fig_sc)
    plt.close(fig_sc)


# -------------------- VIEW 3: FACE RECOGNITION STUDIO --------------------


def render_prediction_view(
    df_people: pd.DataFrame,
    df_filtered: pd.DataFrame,
    width: int,
    height: int,
    n_components: int,
) -> None:
    pretrained = load_pretrained_model(str(MODEL_PATH))

    top_bar_l, top_bar_r = st.columns([1.4, 1.6])
    with top_bar_l:
        model_options = ["Pre-trained PCA Model (.pkl)"] if pretrained is not None else []
        model_options.append("Live Custom PCA + SVM")
        model_source = st.segmented_control(
            "Recognition Engine",
            options=model_options,
            default=model_options[0],
            key="rec_engine",
        )

    if model_source == "Pre-trained PCA Model (.pkl)" and pretrained is not None:
        active_pipeline = pretrained["model"]
        pred_w, pred_h = pretrained["image_size"]
        active_classes = list(pretrained["class_names"])
        model_acc = pretrained["accuracy"]
        k_comp = pretrained["pca_components"]
    else:
        if df_filtered.shape[0] < 2:
            st.warning("Need at least 2 classes in PCA Settings to train a live model.")
            return
        records = list(zip(df_filtered["person"], df_filtered["image_paths"]))
        X, y, _ = load_filtered_faces(records, width, height)
        results = fit_pca_pipeline(X, y, n_components)
        active_pipeline = results["pipeline"]
        pred_w, pred_h = width, height
        active_classes = sorted(df_filtered["person"].tolist())
        model_acc = results["accuracy"]
        k_comp = results["n_components_used"]

    with top_bar_r:
        st.html(
            f"""
            <div style="padding-top: 1.55rem;">
                <span class="pipeline-pill">Input: {pred_w}×{pred_h} Gray</span>
                <span class="pipeline-pill">PCA: {k_comp} Eigenfaces</span>
                <span class="pipeline-pill">Classes: {len(active_classes)}</span>
                <span class="pipeline-pill">Holdout Acc: {model_acc * 100:.1f}%</span>
            </div>
            """
        )

    st.markdown("---")

    # Input selector: 1-Click Visual Gallery, Search by Person, or Upload / Camera
    input_mode = st.segmented_control(
        "Choose how to test a face",
        options=["🎲 1-Click Test Gallery", "🔍 Pick by Person", "📤 Upload / Camera"],
        default="🎲 1-Click Test Gallery",
        key="pred_input_mode",
    )

    person_to_paths = dict(zip(df_people["person"], df_people["image_paths"]))
    eligible_people = [p for p in active_classes if p in person_to_paths]

    if "selected_test_path" not in st.session_state:
        first_person = eligible_people[0] if eligible_people else df_people["person"].iloc[0]
        st.session_state["selected_test_path"] = person_to_paths[first_person][0]
        st.session_state["selected_test_label"] = first_person

    test_image: Image.Image | None = None
    actual_label: str | None = None
    active_test_path: str | None = None

    if input_mode == "🎲 1-Click Test Gallery":
        g_hdr1, g_hdr2 = st.columns([3, 1])
        g_hdr1.caption("Click **Predict** below any portrait to run instant PCA recognition.")
        if "gallery_seed" not in st.session_state:
            st.session_state["gallery_seed"] = 42
        if g_hdr2.button("🔀 Shuffle 6 Faces", width="stretch"):
            st.session_state["gallery_seed"] += 1

        rng = random.Random(st.session_state["gallery_seed"])
        sampled_people = rng.sample(eligible_people, min(6, len(eligible_people)))
        cards = st.columns(len(sampled_people))
        for idx, person in enumerate(sampled_people):
            p_path = rng.choice(person_to_paths[person])
            with cards[idx]:
                st.image(p_path, caption=pretty_name(person), width="stretch")
                if st.button("Predict", key=f"quick_pred_{idx}", width="stretch"):
                    st.session_state["selected_test_path"] = p_path
                    st.session_state["selected_test_label"] = person

        chosen_path = st.session_state.get("selected_test_path")
        if chosen_path and Path(chosen_path).exists():
            test_image = Image.open(chosen_path)
            actual_label = st.session_state.get("selected_test_label")
            active_test_path = chosen_path

    elif input_mode == "🔍 Pick by Person":
        p_col1, p_col2 = st.columns([1.4, 2.6])
        with p_col1:
            chosen_person = st.selectbox(
                "Select known identity",
                eligible_people if eligible_people else df_people["person"].tolist(),
                format_func=pretty_name,
                key="rec_pick_person",
            )
            paths = person_to_paths[chosen_person]
            chosen_idx = st.slider("Photo #", min_value=1, max_value=len(paths), value=1) - 1
            chosen_path = paths[chosen_idx]
            test_image = Image.open(chosen_path)
            actual_label = chosen_person
            active_test_path = chosen_path
        with p_col2:
            st.caption(f"All portraits for {pretty_name(chosen_person)} (showing up to 6)")
            thumbs = st.columns(min(6, len(paths)))
            for i, p in enumerate(paths[:6]):
                thumbs[i].image(p, caption=f"#{i + 1}", width="stretch")

    else:
        up_col1, up_col2 = st.columns(2)
        with up_col1:
            uploaded = st.file_uploader("Upload a portrait (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
        with up_col2:
            use_cam = st.toggle("Enable webcam capture", value=False)
            cam_photo = st.camera_input("Take a snapshot") if use_cam else None

        active_upload = cam_photo if cam_photo is not None else uploaded
        if active_upload is not None:
            test_image = Image.open(active_upload)
            actual_label = None

    if test_image is None:
        st.info("Select or upload a face portrait above to run PCA recognition.")
        return

    # Run PCA projection + SVM prediction
    gray_resized = test_image.convert("L").resize((pred_w, pred_h))
    arr = np.asarray(gray_resized, dtype=np.float32) / 255.0
    x_flat = arr.flatten().reshape(1, -1)

    active_pca: PCA = active_pipeline.named_steps["pca"]
    x_pca = active_pca.transform(x_flat)
    x_recon = np.clip(active_pca.inverse_transform(x_pca).reshape((pred_h, pred_w)), 0, 1)

    candidates = rank_candidates(active_pipeline, x_flat, top_k=5)
    pred_label = candidates[0]["person"]
    top_pct = candidates[0]["pct"]

    # Nearest neighbors in PCA space
    Z_gallery, g_labels, g_paths = compute_pca_gallery_index(
        active_pca,
        tuple(active_classes),
        person_to_paths,
        pred_w,
        pred_h,
    )
    dists = np.linalg.norm(Z_gallery - x_pca, axis=1)
    nn_order = np.argsort(dists)
    # Exclude exact same file path if present
    nn_indices = [
        int(i)
        for i in nn_order
        if active_test_path is None or Path(g_paths[i]) != Path(active_test_path)
    ][:3]

    st.markdown("### Recognition Result")
    res_left, res_mid, res_right = st.columns([1.15, 1.15, 1.5])

    with res_left:
        st.markdown("**1. Query & PCA X-Ray**")
        q1, q2 = st.columns(2)
        q1.image(test_image, caption="Input Portrait", width="stretch")
        q2.image(x_recon, caption=f"PCA Reconstructed\n({k_comp} comps)", width="stretch")
        st.caption(f"Preprocessed to `{pred_w}×{pred_h}` grayscale → projected to `{k_comp}D` eigenface space.")

    with res_mid:
        st.markdown("**2. Predicted Match Reference**")
        ref_paths = person_to_paths.get(pred_label, [])
        if ref_paths:
            ref_idx = 1 if (actual_label == pred_label and len(ref_paths) > 1) else 0
            st.image(
                Image.open(ref_paths[ref_idx]),
                caption=f"Dataset Reference: {pretty_name(pred_label)}",
                width="stretch",
            )

    with res_right:
        is_correct = actual_label is not None and pred_label == actual_label
        is_wrong = actual_label is not None and pred_label != actual_label
        verdict_cls = "verdict-box mismatch" if is_wrong else "verdict-box"
        if is_correct:
            status_text = f"✓ Exact Match (Ground truth: {pretty_name(actual_label)})"
        elif is_wrong:
            status_text = f"✗ Mismatch — Actual person: {pretty_name(actual_label)}"
        else:
            status_text = "Uploaded / Camera Portrait"

        st.html(
            f"""
            <div class="{verdict_cls}">
                <div class="verdict-kicker">Predicted Identity</div>
                <div class="verdict-name">{html.escape(pretty_name(pred_label))}</div>
                <div class="verdict-meta">{html.escape(status_text)} · Relative Score: <strong>{top_pct:.1f}%</strong></div>
            </div>
            """
        )

        st.markdown("**Top 5 Candidate Rankings**")
        bars_html = ""
        for item in candidates:
            name_disp = html.escape(pretty_name(item["person"]))
            pct = max(2.0, min(100.0, item["pct"]))
            bars_html += (
                f'<div class="score-row">'
                f'<div class="score-header"><span>{name_disp}</span><span>{item["pct"]:.1f}%</span></div>'
                f'<div class="score-track"><div class="score-fill" style="width:{pct:.1f}%"></div></div>'
                f"</div>"
            )
        st.html(bars_html)

    st.markdown("**Closest Faces in PCA Eigenface Space (Nearest Neighbors)**")
    nn_cols = st.columns(3)
    for col_i, nn_idx in enumerate(nn_indices):
        with nn_cols[col_i]:
            st.image(
                Image.open(g_paths[nn_idx]),
                caption=f"#{col_i + 1}: {pretty_name(g_labels[nn_idx])} (dist = {dists[nn_idx]:.2f})",
                width=165,
            )


# -------------------- APP ENTRYPOINT --------------------


def main() -> None:
    st.markdown(STUDIO_CSS, unsafe_allow_html=True)

    if not DATASET_DIR.exists():
        st.error(f"Dataset folder not found at `{DATASET_DIR}`.")
        st.stop()

    df_people = index_dataset(str(DATASET_DIR))
    if df_people.empty:
        st.error("No person folders or images found in `lfw_funneled`.")
        st.stop()

    st.html(
        f"""
        <div class="hero-card">
            <div class="hero-badge">PCA · Eigenfaces · LFW Funneled</div>
            <div class="hero-title">EigenStudio — Face EDA &amp; Recognition</div>
            <p class="hero-sub">
                Explore <strong>{int(df_people['image_count'].sum()):,}</strong> portraits across
                <strong>{len(df_people):,}</strong> identities, dissect eigenface compression with PCA,
                and identify faces interactively.
            </p>
        </div>
        """
    )

    nav_col, cfg_col = st.columns([4.2, 1], vertical_alignment="center")
    with nav_col:
        active_page = st.segmented_control(
            "Workspace View",
            options=["📊 Dataset Studio", "🧬 Eigenface Lab", "🎯 Recognition Studio"],
            default="📊 Dataset Studio",
            label_visibility="collapsed",
            width="stretch",
        )
    with cfg_col:
        with st.popover("⚙️ PCA Settings", width="stretch"):
            min_faces = st.slider("Min faces / person", min_value=5, max_value=100, value=20, step=5)
            img_width = st.number_input("Face width (px)", min_value=24, max_value=128, value=50, step=1)
            img_height = st.number_input("Face height (px)", min_value=24, max_value=128, value=37, step=1)
            n_components = st.slider("PCA components", min_value=10, max_value=250, value=150, step=10)

    df_filtered = df_people[df_people["image_count"] >= min_faces].reset_index(drop=True)

    if active_page == "🧬 Eigenface Lab":
        render_pca_view(df_filtered, int(img_width), int(img_height), int(n_components))
    elif active_page == "🎯 Recognition Studio":
        render_prediction_view(df_people, df_filtered, int(img_width), int(img_height), int(n_components))
    else:
        render_eda_view(df_people, df_filtered, int(min_faces))


if __name__ == "__main__":
    main()
