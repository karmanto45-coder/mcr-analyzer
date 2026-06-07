import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

st.set_page_config(
    page_title="MCR-ALS Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a3142; }
[data-testid="stSidebar"] * { color: #c8d0e0 !important; }

.app-header {
    background: linear-gradient(135deg, #0f1117 0%, #161b27 100%);
    border: 1px solid #2a3142;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}
.app-title {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #e2e8f0;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-sub { color: #64748b; font-size: 0.85rem; margin: 4px 0 0; }

.metric-card {
    background: #161b27;
    border: 1px solid #2a3142;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-value { font-family: 'DM Mono', monospace; font-size: 1.6rem; font-weight: 500; color: #7dd3fc; }
.metric-label { font-size: 0.75rem; color: #64748b; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.05em; }

.match-card {
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.5rem;
    border-left: 3px solid;
}
.match-strong { background: #0d2018; border-color: #22c55e; }
.match-medium { background: #1a1a08; border-color: #eab308; }
.match-weak   { background: #1a0a08; border-color: #ef4444; }
.match-conflict { background: #12100d; border-color: #f97316; }

.match-name { font-weight: 500; color: #e2e8f0; font-size: 0.95rem; }
.match-scores { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #94a3b8; margin-top: 3px; }
.match-badge {
    display: inline-block;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 500;
    float: right;
}
.badge-strong { background: #14532d; color: #86efac; }
.badge-medium { background: #422006; color: #fde68a; }
.badge-weak   { background: #450a0a; color: #fca5a5; }
.badge-conflict { background: #431407; color: #fdba74; }

.window-chip {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #7dd3fc;
    margin-right: 6px;
}

.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.5rem 0 0.75rem;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e293b;
}
stButton > button {
    font-family: 'DM Sans', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

LIBRARY_FILE = "spectral_library.json"

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_library():
    if os.path.exists(LIBRARY_FILE):
        with open(LIBRARY_FILE) as f:
            return json.load(f)
    return []

def save_library(lib):
    with open(LIBRARY_FILE, "w") as f:
        json.dump(lib, f, indent=2)

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def hqi(a, b):
    c = cosine_similarity(a, b)
    return round(c ** 2 * 100, 2)

def interpolate_to_grid(wn_ref, spec_ref, wn_target):
    wn_ref = np.array(wn_ref)
    spec_ref = np.array(spec_ref)
    wn_target = np.array(wn_target)
    # np.interp requires xp to be ascending — sort if needed
    if wn_ref[0] > wn_ref[-1]:
        wn_ref = wn_ref[::-1]
        spec_ref = spec_ref[::-1]
    wn_t_sorted = wn_target.copy()
    ascending = wn_t_sorted[0] < wn_t_sorted[-1]
    if not ascending:
        wn_t_sorted = wn_t_sorted[::-1]
    result = np.interp(wn_t_sorted, wn_ref, spec_ref)
    if not ascending:
        result = result[::-1]
    return result

def apply_window(wn, spec, mode, custom_min=None, custom_max=None):
    wn = np.array(wn)
    spec = np.array(spec)
    if mode == "Fingerprint (400–1800 cm⁻¹)":
        mask = (wn >= 400) & (wn <= 1800)
    elif mode == "Full range":
        mask = np.ones(len(wn), dtype=bool)
    elif mode == "Custom range":
        mask = (wn >= custom_min) & (wn <= custom_max)
    else:
        mask = np.ones(len(wn), dtype=bool)
    return wn[mask], spec[mask]

def read_data_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    elif name.endswith((".txt", ".jdx", ".dx")):
        df = pd.read_csv(uploaded, sep=None, engine="python", comment="#")
    else:
        df = pd.read_csv(uploaded)
    return df

def run_mcr_als(D, n_components, max_iter=200, tol=1e-6):
    """MCR-ALS with non-negativity constraints."""
    D = np.array(D, dtype=float)
    m, n = D.shape
    # Init via PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(D)
    C = np.abs(scores)
    C = np.maximum(C, 0)

    lof_history = []
    S = None
    for iteration in range(max_iter):
        # Least squares: S = (CᵀC)⁻¹CᵀD
        try:
            S = np.linalg.lstsq(C, D, rcond=None)[0]
        except Exception:
            break
        S = np.maximum(S, 0)  # non-negativity

        # Least squares: C = DSᵀ(SSᵀ)⁻¹
        try:
            C = np.linalg.lstsq(S.T, D.T, rcond=None)[0].T
        except Exception:
            break
        C = np.maximum(C, 0)  # non-negativity

        D_hat = C @ S
        residual = D - D_hat
        lof = np.sqrt(np.sum(residual**2) / np.sum(D**2)) * 100
        lof_history.append(lof)

        if iteration > 0 and abs(lof_history[-2] - lof_history[-1]) < tol:
            break

    D_hat = C @ S
    r2 = 1 - np.sum((D - D_hat)**2) / np.sum((D - np.mean(D))**2)
    return C, S, lof_history, r2

def detect_components_pca(D, max_k=8):
    pca = PCA(n_components=min(max_k, min(D.shape)-1))
    pca.fit(D)
    ev = pca.explained_variance_ratio_ * 100
    return ev

def match_library(spectrum, wn_sample, library, window_mode, custom_min, custom_max, top_n=10):
    results = []
    wn_s, spec_s = apply_window(wn_sample, spectrum, window_mode, custom_min, custom_max)
    for entry in library:
        wn_r = np.array(entry["wavenumber"])
        sp_r = np.array(entry["spectrum"])
        wn_r2, sp_r2 = apply_window(wn_r, sp_r, window_mode, custom_min, custom_max)
        if len(wn_r2) < 5 or len(wn_s) < 5:
            continue
        sp_interp = interpolate_to_grid(wn_r2, sp_r2, wn_s)
        cos = round(cosine_similarity(spec_s, sp_interp), 4)
        hq  = round(hqi(spec_s, sp_interp), 2)
        results.append({
            "name": entry["name"],
            "category": entry.get("category", "—"),
            "cosine": cos,
            "hqi": hq,
            "added": entry.get("added", "—")
        })
    results.sort(key=lambda x: x["cosine"], reverse=True)
    return results[:top_n]

def consensus_status(cos, hq):
    cos_ok  = cos >= 0.95
    hqi_ok  = hq  >= 90.25   # 0.95² × 100
    cos_med = cos >= 0.90
    hqi_med = hq  >= 81.0
    if cos_ok and hqi_ok:
        return "match-strong", "badge-strong", "Match kuat"
    if cos_med and hqi_med:
        return "match-medium", "badge-medium", "Match sedang"
    if (cos_ok and not hqi_ok) or (hqi_ok and not cos_ok):
        return "match-conflict", "badge-conflict", "Ranking konflik"
    return "match-weak", "badge-weak", "Tidak match"

# ── App shell ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <p class="app-title">MCR-ALS Analyzer</p>
  <p class="app-sub">Multivariate Curve Resolution · ATR-FTIR · Spectral Matching</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Input Data",
    "🔬 Analisis MCR",
    "🔍 Spectral Matching",
    "📚 Library Acuan",
    "📊 Laporan"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — INPUT DATA
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Upload data spektra</p>', unsafe_allow_html=True)

    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload file spektra (Excel / CSV / TXT)",
            type=["xlsx", "xls", "csv", "txt", "jdx", "dx"],
            help="Kolom pertama = wavenumber, kolom berikutnya = tiap spektra"
        )
    with col_info:
        st.info("""
**Format yang didukung**
- Excel (.xlsx, .xls)
- CSV (.csv)
- Text (.txt)
- JCAMP-DX (.jdx, .dx)

Kolom 1 = wavenumber (cm⁻¹)
Kolom 2+ = spektra sampel
        """)

    if uploaded:
        try:
            df = read_data_file(uploaded)
            st.markdown('<p class="section-header">Pratinjau data</p>', unsafe_allow_html=True)

            # Deteksi kolom wavenumber
            wn_col = df.columns[0]
            spec_cols = df.columns[1:]
            wavenumber = df[wn_col].values
            spectra_matrix = df[spec_cols].values

            n_spec = len(spec_cols)
            n_points = len(wavenumber)
            wn_min = float(wavenumber.min())
            wn_max = float(wavenumber.max())

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{n_spec}</div><div class="metric-label">Jumlah spektra</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{n_points}</div><div class="metric-label">Titik data</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{wn_min:.0f}</div><div class="metric-label">Wavenumber min</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{wn_max:.0f}</div><div class="metric-label">Wavenumber max</div></div>', unsafe_allow_html=True)

            st.markdown('<p class="section-header">Pra-pemrosesan</p>', unsafe_allow_html=True)
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                do_norm = st.checkbox("Normalisasi (area = 1)", value=True)
            with col_p2:
                do_smooth = st.checkbox("Smoothing (Savitzky-Golay)", value=False)
            with col_p3:
                do_baseline = st.checkbox("Koreksi baseline (min subtraction)", value=False)

            proc_spectra = spectra_matrix.copy().astype(float)
            if do_smooth:
                for i in range(proc_spectra.shape[1]):
                    proc_spectra[:, i] = savgol_filter(proc_spectra[:, i], 11, 3)
            if do_baseline:
                for i in range(proc_spectra.shape[1]):
                    proc_spectra[:, i] -= proc_spectra[:, i].min()
            if do_norm:
                for i in range(proc_spectra.shape[1]):
                  area = np.trapezoid(np.abs(proc_spectra[:, i]), wavenumber)
                    if area > 0:
                        proc_spectra[:, i] /= area

            # Simpan ke session
            st.session_state["wavenumber"] = wavenumber
            st.session_state["spectra"] = proc_spectra
            st.session_state["spec_names"] = list(spec_cols)

            # Plot
            st.markdown('<p class="section-header">Visualisasi spektra</p>', unsafe_allow_html=True)
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, col in enumerate(spec_cols):
                fig.add_trace(go.Scatter(
                    x=wavenumber, y=proc_spectra[:, i],
                    name=str(col), mode="lines",
                    line=dict(width=1.2, color=colors[i % len(colors)])
                ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f1117",
                plot_bgcolor="#0f1117",
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title="Absorbance",
                xaxis=dict(autorange="reversed", gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b"),
                legend=dict(bgcolor="#161b27", bordercolor="#2a3142", borderwidth=1),
                margin=dict(l=20, r=20, t=20, b=40),
                height=380
            )
            st.plotly_chart(fig, use_container_width=True)

            if n_spec < 4:
                st.warning(f"⚠️ Hanya {n_spec} spektra terdeteksi. Minimum yang direkomendasikan adalah 4 spektra (kondisi ekstrim). Hasil MCR mungkin tidak stabil.")
            else:
                st.success(f"✅ {n_spec} spektra siap dianalisis.")

        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    else:
        st.markdown("""
        <div style="border:1px dashed #2a3142;border-radius:10px;padding:3rem;text-align:center;color:#475569;">
            Upload file spektra di atas untuk memulai
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — ANALISIS MCR
# ════════════════════════════════════════════════════════════════
with tab2:
    if "spectra" not in st.session_state:
        st.info("Upload data spektra di tab 'Input Data' terlebih dahulu.")
    else:
        wn = st.session_state["wavenumber"]
        D  = st.session_state["spectra"].T  # shape: n_spectra × n_points

        st.markdown('<p class="section-header">Deteksi jumlah komponen (PCA)</p>', unsafe_allow_html=True)
        ev = detect_components_pca(D)
        cumev = np.cumsum(ev)

        fig_pca = make_subplots(rows=1, cols=2,
            subplot_titles=("Variansi tiap komponen (%)", "Variansi kumulatif (%)"))
        fig_pca.add_trace(go.Bar(x=list(range(1, len(ev)+1)), y=ev,
            marker_color="#7dd3fc", name="Variansi"), row=1, col=1)
        fig_pca.add_trace(go.Scatter(x=list(range(1, len(cumev)+1)), y=cumev,
            mode="lines+markers", line=dict(color="#f97316"), name="Kumulatif"), row=1, col=2)
        fig_pca.add_hline(y=95, line_dash="dash", line_color="#475569",
            annotation_text="95%", row=1, col=2)
        fig_pca.update_layout(template="plotly_dark", paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117", height=280, showlegend=False,
            margin=dict(l=20,r=20,t=40,b=20))
        fig_pca.update_xaxes(gridcolor="#1e293b")
        fig_pca.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig_pca, use_container_width=True)

        # Auto-suggest
        auto_k = int(np.searchsorted(cumev, 95)) + 1
        auto_k = max(2, min(auto_k, len(ev)))
        st.caption(f"Saran otomatis PCA: **{auto_k} komponen** menjelaskan ≥ 95% variansi")

        st.markdown('<p class="section-header">Parameter MCR-ALS</p>', unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            n_comp = st.number_input("Jumlah komponen", min_value=2, max_value=8, value=auto_k)
        with col_b:
            max_iter = st.number_input("Iterasi maksimum", min_value=50, max_value=1000, value=200, step=50)
        with col_c:
            tol = st.selectbox("Toleransi konvergensi", [1e-4, 1e-5, 1e-6, 1e-7], index=2,
                format_func=lambda x: f"{x:.0e}")

        if st.button("▶  Jalankan MCR-ALS", use_container_width=True):
            with st.spinner("Menjalankan MCR-ALS..."):
                C, S, lof_hist, r2 = run_mcr_als(D, int(n_comp), int(max_iter), float(tol))
                st.session_state["mcr_C"] = C
                st.session_state["mcr_S"] = S
                st.session_state["mcr_lof"] = lof_hist
                st.session_state["mcr_r2"] = r2
                st.session_state["mcr_ncomp"] = int(n_comp)
                st.success(f"✅ MCR-ALS selesai — {len(lof_hist)} iterasi | LOF akhir: {lof_hist[-1]:.4f}% | R²: {r2:.5f}")

        if "mcr_S" in st.session_state:
            S = st.session_state["mcr_S"]
            C = st.session_state["mcr_C"]
            lof_hist = st.session_state["mcr_lof"]
            r2 = st.session_state["mcr_r2"]
            n_comp = st.session_state["mcr_ncomp"]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{lof_hist[-1]:.3f}%</div><div class="metric-label">LOF akhir</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{r2:.4f}</div><div class="metric-label">R²</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{len(lof_hist)}</div><div class="metric-label">Iterasi</div></div>', unsafe_allow_html=True)

            st.markdown('<p class="section-header">Spektra murni hasil MCR</p>', unsafe_allow_html=True)
            fig_s = go.Figure()
            colors = px.colors.qualitative.Pastel
            for i in range(n_comp):
                fig_s.add_trace(go.Scatter(
                    x=wn, y=S[i],
                    name=f"Komponen {i+1}",
                    mode="lines",
                    line=dict(width=1.8, color=colors[i % len(colors)])
                ))
            fig_s.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Intensitas",
                xaxis=dict(autorange="reversed", gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b"),
                legend=dict(bgcolor="#161b27"), height=350,
                margin=dict(l=20,r=20,t=20,b=40)
            )
            st.plotly_chart(fig_s, use_container_width=True)

            st.markdown('<p class="section-header">Profil konsentrasi</p>', unsafe_allow_html=True)
            spec_names = st.session_state.get("spec_names", [f"S{i+1}" for i in range(C.shape[0])])
            fig_c = go.Figure()
            for i in range(n_comp):
                fig_c.add_trace(go.Bar(
                    name=f"Komponen {i+1}",
                    x=spec_names, y=C[:, i],
                    marker_color=colors[i % len(colors)]
                ))
            fig_c.update_layout(
                barmode="stack", template="plotly_dark",
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis_title="Sampel", yaxis_title="Kontribusi relatif",
                xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                legend=dict(bgcolor="#161b27"), height=300,
                margin=dict(l=20,r=20,t=20,b=40)
            )
            st.plotly_chart(fig_c, use_container_width=True)

            st.markdown('<p class="section-header">Konvergensi LOF</p>', unsafe_allow_html=True)
            fig_lof = go.Figure()
            fig_lof.add_trace(go.Scatter(y=lof_hist, mode="lines",
                line=dict(color="#f97316", width=1.5)))
            fig_lof.update_layout(
                template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis_title="Iterasi", yaxis_title="LOF (%)",
                xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                height=220, margin=dict(l=20,r=20,t=20,b=40)
            )
            st.plotly_chart(fig_lof, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — SPECTRAL MATCHING
# ════════════════════════════════════════════════════════════════
with tab3:
    if "mcr_S" not in st.session_state:
        st.info("Jalankan MCR-ALS di tab 'Analisis MCR' terlebih dahulu.")
    else:
        library = load_library()
        S = st.session_state["mcr_S"]
        wn = st.session_state["wavenumber"]
        n_comp = st.session_state["mcr_ncomp"]

        if len(library) == 0:
            st.warning("Library acuan masih kosong. Tambahkan spektra referensi di tab 'Library Acuan'.")
        else:
            st.markdown('<p class="section-header">Pengaturan window spektra</p>', unsafe_allow_html=True)

            col_w1, col_w2, col_w3 = st.columns([2, 1, 1])
            with col_w1:
                window_mode = st.selectbox(
                    "Rentang analisis kemiripan",
                    ["Fingerprint (400–1800 cm⁻¹)", "Full range", "Custom range"],
                    help="Pilih rentang wavenumber yang digunakan untuk menghitung kemiripan"
                )
            with col_w2:
                custom_min = st.number_input("Min (cm⁻¹)", value=400, step=50,
                    disabled=(window_mode != "Custom range"))
            with col_w3:
                custom_max = st.number_input("Max (cm⁻¹)", value=4000, step=50,
                    disabled=(window_mode != "Custom range"))

            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                top_n = st.number_input("Tampilkan Top-N kandidat", min_value=3, max_value=20, value=10)
            with col_t2:
                thresh_cos = st.slider("Threshold cosine similarity", 0.70, 1.00, 0.90, 0.01)
            with col_t3:
                thresh_hqi = st.slider("Threshold HQI (%)", 50.0, 100.0, 81.0, 0.5)

            # Tampilkan window aktif
            wn_arr = np.array(wn)
            if window_mode == "Fingerprint (400–1800 cm⁻¹)":
                wmin_show, wmax_show = 400, 1800
            elif window_mode == "Custom range":
                wmin_show, wmax_show = custom_min, custom_max
            else:
                wmin_show, wmax_show = float(wn_arr.min()), float(wn_arr.max())

            st.markdown(
                f'<span class="window-chip">{wmin_show:.0f} – {wmax_show:.0f} cm⁻¹</span>'
                f'<span style="font-size:0.8rem;color:#475569">{window_mode} · {int(np.sum((wn_arr>=wmin_show)&(wn_arr<=wmax_show)))} titik data</span>',
                unsafe_allow_html=True
            )

            st.markdown('<p class="section-header">Hasil matching per komponen</p>', unsafe_allow_html=True)

            for i in range(n_comp):
                with st.expander(f"Komponen {i+1}", expanded=(i == 0)):
                    results = match_library(
                        S[i], wn, library, window_mode,
                        custom_min, custom_max, int(top_n)
                    )

                    if not results:
                        st.warning("Tidak ada hasil — periksa rentang wavenumber library vs sampel.")
                        continue

                    # Plot overlay spektra MCR vs top match
                    top = results[0]
                    lib_entry = next((e for e in library if e["name"] == top["name"]), None)
                    if lib_entry:
                        fig_ov = go.Figure()
                        fig_ov.add_trace(go.Scatter(
                            x=wn, y=S[i], name=f"Komponen {i+1} (MCR)",
                            line=dict(color="#7dd3fc", width=1.8)
                        ))
                        wn_r = np.array(lib_entry["wavenumber"])
                        sp_r = np.array(lib_entry["spectrum"])
                        sp_interp = interpolate_to_grid(wn_r, sp_r, wn)
                        # Normalize for overlay display
                        if S[i].max() > 0 and sp_interp.max() > 0:
                            sp_disp = sp_interp / sp_interp.max() * S[i].max()
                        else:
                            sp_disp = sp_interp
                        fig_ov.add_trace(go.Scatter(
                            x=wn, y=sp_disp, name=top["name"],
                            line=dict(color="#f97316", width=1.5, dash="dot")
                        ))
                        # Shade matching window
                        fig_ov.add_vrect(
                            x0=wmin_show, x1=wmax_show,
                            fillcolor="#7dd3fc", opacity=0.04,
                            annotation_text="window", annotation_position="top left"
                        )
                        fig_ov.update_layout(
                            template="plotly_dark", paper_bgcolor="#0f1117",
                            plot_bgcolor="#0f1117",
                            xaxis=dict(autorange="reversed", gridcolor="#1e293b",
                                       title="Wavenumber (cm⁻¹)"),
                            yaxis=dict(gridcolor="#1e293b", title="Intensitas (norm.)"),
                            legend=dict(bgcolor="#161b27"),
                            height=280, margin=dict(l=20,r=20,t=20,b=40)
                        )
                        st.plotly_chart(fig_ov, use_container_width=True)

                    # Ranking tabel
                    for rank, r in enumerate(results, 1):
                        card_cls, badge_cls, badge_txt = consensus_status(r["cosine"], r["hqi"])
                        cos_flag = "✓" if r["cosine"] >= thresh_cos else "✗"
                        hqi_flag = "✓" if r["hqi"] >= thresh_hqi else "✗"
                        rank_conflict = ""
                        if rank > 1:
                            prev = results[rank-2]
                            if (r["cosine"] > prev["cosine"]) != (r["hqi"] > prev["hqi"]):
                                rank_conflict = " · ⚠ ranking konflik"

                        st.markdown(f"""
                        <div class="match-card {card_cls}">
                          <span class="match-badge {badge_cls}">{badge_txt}</span>
                          <span class="match-name">#{rank} &nbsp; {r['name']}</span>
                          <div class="match-scores">
                            Cosine: <b>{r['cosine']:.4f}</b> {cos_flag} &nbsp;|&nbsp;
                            HQI: <b>{r['hqi']:.2f}%</b> {hqi_flag} &nbsp;|&nbsp;
                            Kategori: {r['category']}{rank_conflict}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 4 — LIBRARY ACUAN
# ════════════════════════════════════════════════════════════════
with tab4:
    library = load_library()

    st.markdown('<p class="section-header">Tambah spektra acuan</p>', unsafe_allow_html=True)

    with st.form("add_ref"):
        col_n, col_cat = st.columns(2)
        with col_n:
            ref_name = st.text_input("Nama senyawa / komponen", placeholder="mis. Asam Oleat")
        with col_cat:
            ref_cat = st.text_input("Kategori", placeholder="mis. Asam lemak, Polimer, ...")

        ref_file = st.file_uploader("Upload spektra acuan (Excel / CSV)", type=["xlsx","xls","csv","txt"])
        ref_notes = st.text_area("Catatan (opsional)", height=60)

        submitted = st.form_submit_button("Tambahkan ke Library")
        if submitted:
            if not ref_name:
                st.error("Nama senyawa wajib diisi.")
            elif ref_file is None:
                st.error("File spektra wajib diupload.")
            else:
                try:
                    df_r = read_data_file(ref_file)
                    wn_r = df_r.iloc[:, 0].values.tolist()
                    sp_r = df_r.iloc[:, 1].values.tolist()
                    entry = {
                        "name": ref_name,
                        "category": ref_cat,
                        "notes": ref_notes,
                        "wavenumber": wn_r,
                        "spectrum": sp_r,
                        "added": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    library.append(entry)
                    save_library(library)
                    st.success(f"✅ '{ref_name}' berhasil ditambahkan ke library.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal membaca file: {e}")

    # Tampilkan isi library
    st.markdown(f'<p class="section-header">Library acuan — {len(library)} entri</p>', unsafe_allow_html=True)

    if library:
        # Import batch
        with st.expander("Import library dari file JSON"):
            imp_file = st.file_uploader("Upload file library (.json)", type=["json"], key="imp")
            if imp_file:
                imported = json.load(imp_file)
                existing_names = {e["name"] for e in library}
                new_entries = [e for e in imported if e["name"] not in existing_names]
                st.write(f"{len(imported)} entri ditemukan · {len(new_entries)} baru")
                if st.button("Merge ke library"):
                    library.extend(new_entries)
                    save_library(library)
                    st.success(f"✅ {len(new_entries)} entri baru ditambahkan.")
                    st.rerun()

        # Export
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            lib_json = json.dumps(library, indent=2)
            st.download_button("⬇ Export library (JSON)", lib_json,
                "spectral_library.json", "application/json")

        # List entri
        for idx, entry in enumerate(library):
            with st.container():
                col_info, col_del = st.columns([5, 1])
                with col_info:
                    st.markdown(f"""
                    <div style="background:#161b27;border:1px solid #2a3142;border-radius:8px;
                                padding:0.75rem 1rem;margin-bottom:6px;">
                      <span style="font-weight:500;color:#e2e8f0;">{entry['name']}</span>
                      <span style="font-size:0.75rem;color:#475569;margin-left:10px;">{entry.get('category','—')}</span>
                      <span style="font-size:0.75rem;color:#334155;float:right;">{entry.get('added','—')}</span>
                      <div style="font-size:0.75rem;color:#64748b;margin-top:2px;">
                        {len(entry['wavenumber'])} titik · 
                        {min(entry['wavenumber']):.0f}–{max(entry['wavenumber']):.0f} cm⁻¹
                        {(' · ' + entry['notes']) if entry.get('notes') else ''}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_del:
                    if st.button("✕", key=f"del_{idx}", help="Hapus entri ini"):
                        library.pop(idx)
                        save_library(library)
                        st.rerun()
    else:
        st.markdown("""
        <div style="border:1px dashed #2a3142;border-radius:10px;padding:3rem;
                    text-align:center;color:#475569;">
            Library masih kosong — tambahkan spektra acuan di atas
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 5 — LAPORAN
# ════════════════════════════════════════════════════════════════
with tab5:
    if "mcr_S" not in st.session_state:
        st.info("Jalankan MCR-ALS terlebih dahulu untuk menghasilkan laporan.")
    else:
        S = st.session_state["mcr_S"]
        C = st.session_state["mcr_C"]
        wn = st.session_state["wavenumber"]
        lof_hist = st.session_state["mcr_lof"]
        r2 = st.session_state["mcr_r2"]
        n_comp = st.session_state["mcr_ncomp"]
        spec_names = st.session_state.get("spec_names", [])

        st.markdown('<p class="section-header">Ringkasan hasil MCR-ALS</p>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{n_comp}</div><div class="metric-label">Komponen</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{lof_hist[-1]:.3f}%</div><div class="metric-label">LOF akhir</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{r2:.4f}</div><div class="metric-label">R²</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(lof_hist)}</div><div class="metric-label">Iterasi</div></div>', unsafe_allow_html=True)

        st.markdown('<p class="section-header">Export data</p>', unsafe_allow_html=True)

        col_e1, col_e2, col_e3 = st.columns(3)

        # Export spektra murni
        with col_e1:
            df_S = pd.DataFrame(S.T, index=wn,
                columns=[f"Komponen_{i+1}" for i in range(n_comp)])
            df_S.index.name = "Wavenumber"
            st.download_button(
                "⬇ Spektra murni (CSV)",
                df_S.to_csv(),
                "mcr_pure_spectra.csv",
                "text/csv",
                use_container_width=True
            )

        # Export profil konsentrasi
        with col_e2:
            df_C = pd.DataFrame(C,
                index=spec_names if spec_names else range(C.shape[0]),
                columns=[f"Komponen_{i+1}" for i in range(n_comp)])
            df_C.index.name = "Sampel"
            st.download_button(
                "⬇ Profil konsentrasi (CSV)",
                df_C.to_csv(),
                "mcr_concentration.csv",
                "text/csv",
                use_container_width=True
            )

        # Export LOF history
        with col_e3:
            df_lof = pd.DataFrame({"Iterasi": range(1, len(lof_hist)+1), "LOF (%)": lof_hist})
            st.download_button(
                "⬇ LOF history (CSV)",
                df_lof.to_csv(index=False),
                "mcr_lof_history.csv",
                "text/csv",
                use_container_width=True
            )

        # Laporan teks
        st.markdown('<p class="section-header">Laporan teks</p>', unsafe_allow_html=True)
        report = f"""MCR-ALS ANALYZER — LAPORAN ANALISIS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

PARAMETER ANALISIS
  Jumlah komponen : {n_comp}
  Iterasi         : {len(lof_hist)}
  LOF akhir       : {lof_hist[-1]:.4f} %
  R²              : {r2:.6f}

SPEKTRA MURNI
  Wavenumber range: {float(np.array(wn).min()):.1f} – {float(np.array(wn).max()):.1f} cm⁻¹
  Jumlah titik    : {len(wn)}

PROFIL KONSENTRASI
{'  Sampel':20s} {'Komp.1':>8} {'Komp.2':>8} {'Komp.3':>8}
"""
        for i, row in enumerate(C):
            nm = spec_names[i] if i < len(spec_names) else f"S{i+1}"
            vals = "  ".join([f"{v:8.4f}" for v in row])
            report += f"  {nm[:20]:20s} {vals}\n"

        st.text_area("", report, height=300)
        st.download_button("⬇ Download laporan (.txt)", report,
            "mcr_report.txt", "text/plain")
