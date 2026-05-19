import os

# Must be set BEFORE any tensorflow import
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF C++ warnings

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
from PIL import Image
import io

# ─────────────────────────────────────────────────────────────────────────────
# TensorFlow (optional — only needed for MRI)
# ─────────────────────────────────────────────────────────────────────────────
TF_AVAILABLE = False
TF_ERROR     = ""
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.applications.densenet import preprocess_input
    TF_AVAILABLE = True
except Exception as _e:
    TF_ERROR = str(_e)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp { background: #050d1a; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid #1a2d45;
}
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 13px;
}

/* ── Hide default Streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1100px; }

/* ── Custom card ── */
.ns-card {
    background: #0d1f35;
    border: 1px solid #1a2d45;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.ns-card-accent {
    background: linear-gradient(135deg, #0d1f35 0%, #0a1a2e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
}

/* ── Section title ── */
.ns-section-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.ns-title {
    font-size: 22px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 6px;
}
.ns-subtitle {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 20px;
    line-height: 1.55;
}

/* ── Risk meter ── */
.risk-bar-container {
    background: #0a1628;
    border-radius: 8px;
    height: 14px;
    overflow: hidden;
    border: 1px solid #1a2d45;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s ease;
}

/* ── Result box ── */
.result-high {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 12px;
    padding: 20px 24px;
}
.result-moderate {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 12px;
    padding: 20px 24px;
}
.result-low {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 12px;
    padding: 20px 24px;
}

/* ── Metric pill ── */
.metric-pill {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 12px;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}

/* ── Stacked metric ── */
.ns-metric {
    background: #0a1628;
    border: 1px solid #1a2d45;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.ns-metric .val {
    font-size: 24px;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1;
}
.ns-metric .lbl {
    font-size: 11px;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Model status dot ── */
.dot-on  { display:inline-block; width:7px; height:7px; border-radius:50%; background:#34d399; margin-right:6px; }
.dot-off { display:inline-block; width:7px; height:7px; border-radius:50%; background:#ef4444; margin-right:6px; }

/* ── Input labels ── */
.stTextInput label, .stNumberInput label, .stSelectbox label, .stFileUploader label {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
    letter-spacing: 0.03em !important;
}
.stButton > button:hover {
    filter: brightness(1.1) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.35) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0a1628 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid #1a2d45 !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    color: #64748b !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #38bdf8 !important;
}

/* ── Number inputs ── */
input[type="number"] {
    background: #0a1628 !important;
    border: 1px solid #1a2d45 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Divider ── */
.ns-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1a2d45, transparent);
    margin: 20px 0;
}

/* ── Warning banner ── */
.ns-warning {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 10px;
    padding: 12px 16px;
    color: #fbbf24;
    font-size: 13px;
}
.ns-info {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    color: #7dd3fc;
    font-size: 13px;
}

/* ── Sidebar nav items ── */
.nav-item {
    padding: 10px 14px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    color: #94a3b8;
    transition: all 0.15s;
}
.nav-item:hover { background: #1a2d45; color: #e2e8f0; }
.nav-item.active { background: rgba(56,189,248,0.12); color: #38bdf8; border-left: 2px solid #38bdf8; }
</style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    mdls = {}

    # Clinical
    for key, fname in [
        ("clin_model",    "models/clinical_xgb.pkl"),
        ("clin_scaler",   "models/clinical_scaler.pkl"),
        ("clin_features", "models/clinical_features.pkl"),
    ]:
        try:    mdls[key] = joblib.load(fname)
        except: mdls[key] = None

    # Biomarker
    for key, fname in [
        ("bio_model",    "models/biomarker_xgb_model.pkl"),
        ("bio_scaler",   "models/biomarker_scaler.pkl"),
        ("bio_features", "models/biomarker_features.pkl"),
    ]:
        try:    mdls[key] = joblib.load(fname)
        except: mdls[key] = None

    # Genetic
    for key, fname in [
        ("gen_pipeline", "models/gene_xgb_pipeline.pkl"),
        ("gen_genes",    "models/gene_feature_names.pkl"),
    ]:
        try:    mdls[key] = joblib.load(fname)
        except: mdls[key] = None

    # MRI – XGBoost
    try:    mdls["mri_xgb"] = joblib.load("models/mri_xgb_densenet.pkl")
    except: mdls["mri_xgb"] = None

    # MRI – DenseNet121 feature extractor (1024-dim GAP, 224×224 input)
    mdls["mri_extractor"]  = None
    mdls["mri_load_error"] = ""

    if not TF_AVAILABLE:
        mdls["mri_load_error"] = f"TensorFlow not available: {TF_ERROR}"
    else:
        keras_path = "models/mri_densenet121_finetuned.keras"
        if not os.path.exists(keras_path):
            mdls["mri_load_error"] = f"File not found: {os.path.abspath(keras_path)}"
        else:
            def _find_gap_layer(base):
                """Find GlobalAveragePooling2D — the 1024-dim bottleneck layer."""
                # Pass 1: search by layer name — handles 'global_average_pooling2d'
                # and the named 'gap_features' layer used in the fixed notebook
                for layer in reversed(base.layers):
                    name = layer.name.lower()
                    if "global_average_pooling" in name or name == "gap_features":
                        return layer
                # Pass 2: search by output shape == (None, 1024)
                for layer in reversed(base.layers):
                    try:
                        shape = layer.output.shape
                        if len(shape) == 2 and shape[-1] == 1024:
                            return layer
                    except Exception:
                        pass
                # Pass 3: known position in the architecture (GAP is -3 from end)
                return base.layers[-3]

            def _make_extractor(base, KModel=None):
                """Build a sub-model that stops at the 1024-dim GAP layer.
                NEVER falls back to the full model — that would return 4 outputs."""
                if KModel is None:
                    KModel = Model
                gap = _find_gap_layer(base)
                extractor = KModel(inputs=base.input, outputs=gap.output)
                # Verify output shape is (None, 1024) before returning
                out_shape = extractor.output_shape
                if isinstance(out_shape, (list, tuple)) and out_shape[-1] != 1024:
                    raise RuntimeError(
                        f"Extractor output shape is {out_shape} — expected (None, 1024). "
                        f"GAP layer found: '{gap.name}'. "
                        "The wrong layer was selected as the feature bottleneck."
                    )
                return extractor

            errors = []
            h5_path      = "models/mri_densenet121_weights.h5"      # weights-only file
            h5_full_path = keras_path.replace(".keras", ".h5")        # full model H5 (if exists)

            # ── Strategy 0: full model H5 (most compatible across TF/Keras versions)
            # Only works if model was saved with model.save('file.h5') — includes architecture
            if mdls["mri_extractor"] is None:
                try:
                    from tensorflow.keras.models import load_model as _lm
                    if os.path.exists(h5_full_path):
                        base = _lm(h5_full_path, compile=False)
                        mdls["mri_extractor"] = _make_extractor(base)
                        errors.append("0 (H5 full model) succeeded")
                    else:
                        errors.append(f"0 (H5 full model): {h5_full_path} not found")
                except Exception as e:
                    errors.append(f"0 (H5 full model): {e}")

            # ── Strategy 0b: full model H5 via tf_keras
            if mdls["mri_extractor"] is None:
                try:
                    import tf_keras as _tfk0
                    if os.path.exists(h5_full_path):
                        base = _tfk0.models.load_model(h5_full_path, compile=False)
                        mdls["mri_extractor"] = _make_extractor(base, _tfk0.Model)
                        errors.append("0b (H5 full tf_keras) succeeded")
                    else:
                        errors.append(f"0b (H5 full tf_keras): {h5_full_path} not found")
                except Exception as e:
                    errors.append(f"0b (H5 full tf_keras): {e}")
            if mdls["mri_extractor"] is None:
                try:
                    import keras as _keras
                    base = _keras.saving.load_model(keras_path, compile=False)
                    mdls["mri_extractor"] = _make_extractor(base, _keras.Model)
                except Exception as e:
                    errors.append(f"A (keras.saving): {e}")

            # ── Strategy B: tf_keras package (pip install tf_keras)
            if mdls["mri_extractor"] is None:
                try:
                    import tf_keras as _tfk
                    base = _tfk.models.load_model(keras_path, compile=False)
                    mdls["mri_extractor"] = _make_extractor(base, _tfk.Model)
                except Exception as e:
                    errors.append(f"B (tf_keras): {e}")

            # ── Strategy C: plain tf.keras load (works on TF ≤ 2.15 / keras 2)
            if mdls["mri_extractor"] is None:
                try:
                    base = load_model(keras_path, compile=False)
                    mdls["mri_extractor"] = _make_extractor(base)
                except Exception as e:
                    errors.append(f"C (load_model compile=False): {e}")

            # ── Strategy D: safe_mode=False (keras 3 sometimes requires this)
            if mdls["mri_extractor"] is None:
                try:
                    base = load_model(keras_path, compile=False, safe_mode=False)
                    mdls["mri_extractor"] = _make_extractor(base)
                except Exception as e:
                    errors.append(f"D (safe_mode=False): {e}")

            # ── Strategy E: tf.keras.saving alias
            if mdls["mri_extractor"] is None:
                try:
                    base = tf.keras.saving.load_model(keras_path, compile=False)
                    mdls["mri_extractor"] = _make_extractor(base)
                except Exception as e:
                    errors.append(f"E (tf.keras.saving): {e}")

            # ── Strategy F: rebuild architecture + load weights from .keras zip.
            # Explicitly wire the extractor output to the GAP layer (NOT Dense(4)).
            if mdls["mri_extractor"] is None:
                try:
                    import zipfile, tempfile

                    # Prefer tf_keras for rebuild; fall back to tensorflow.keras
                    try:
                        import tf_keras as _tfk2
                        _DenseNet121 = _tfk2.applications.DenseNet121
                        _GAP         = _tfk2.layers.GlobalAveragePooling2D
                        _BN          = _tfk2.layers.BatchNormalization
                        _Dense       = _tfk2.layers.Dense
                        _Dropout     = _tfk2.layers.Dropout
                        _KModel      = _tfk2.Model
                    except ImportError:
                        from tensorflow.keras.applications import DenseNet121 as _DenseNet121
                        from tensorflow.keras.layers import (
                            GlobalAveragePooling2D as _GAP,
                            BatchNormalization as _BN,
                            Dense as _Dense,
                            Dropout as _Dropout,
                        )
                        from tensorflow.keras.models import Model as _KModel

                    # Build the full architecture exactly as in the DenseNet121 notebook
                    base_net   = _DenseNet121(
                        weights=None, include_top=False, input_shape=(224, 224, 3)
                    )
                    gap_output = _GAP(name="gap_features")(base_net.output)   # 1024-dim
                    bn_output  = _BN()(gap_output)
                    dropped    = _Dropout(0.4)(bn_output)
                    out        = _Dense(4, activation="softmax")(dropped)
                    full_model = _KModel(inputs=base_net.input, outputs=out)

                    # Load weights — supports both weights-only H5 and .keras zip
                    if os.path.exists(h5_path):
                        # weights-only file (mri_densenet121_weights.h5)
                        full_model.load_weights(h5_path)
                        errors.append("F used weights-only H5 file")
                    else:
                        # fallback: try to unzip .keras archive for weights
                        import zipfile, tempfile
                        with zipfile.ZipFile(keras_path, "r") as zf:
                            names  = zf.namelist()
                            w_file = next((n for n in names if n.endswith(".weights.h5")), None)
                            if w_file is None:
                                w_file = next((n for n in names if n.endswith(".h5")), None)
                            if not w_file:
                                raise RuntimeError("No .h5 weights file inside .keras archive")
                            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                                tmp.write(zf.read(w_file))
                                tmp_path = tmp.name
                        full_model.load_weights(tmp_path)
                        os.unlink(tmp_path)

                    # Build extractor that outputs the GAP layer (1280-dim), not Dense(4)
                    mdls["mri_extractor"] = _KModel(
                        inputs=full_model.input,
                        outputs=full_model.get_layer(index=-3).output,  # GAP layer
                    )
                    # Sanity-check the output shape
                    if mdls["mri_extractor"].output_shape[-1] != 1024:
                        raise RuntimeError(
                            f"Rebuilt extractor output shape = "
                            f"{mdls['mri_extractor'].output_shape}, expected (None, 1024)"
                        )
                    errors.append("F (architecture rebuild) succeeded")
                except Exception as e:
                    mdls["mri_extractor"] = None
                    errors.append(f"F (rebuild): {e}")

            if mdls["mri_extractor"] is None:
                mdls["mri_load_error"] = (
                    "All loading strategies failed. "
                    "Fix: pip install tf_keras  OR  pip install keras  "
                    "then restart the app.\n\n"
                    "Details: " + " | ".join(errors)
                )

    return mdls

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS – RISK DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
def risk_color(prob):
    if prob >= 0.65:  return "#ef4444"
    if prob >= 0.40:  return "#f59e0b"
    return "#34d399"

def risk_label(prob):
    if prob >= 0.65:  return "High Risk", "result-high"
    if prob >= 0.40:  return "Moderate Risk", "result-moderate"
    return "Low Risk", "result-low"

def render_risk_bar(prob):
    pct   = int(prob * 100)
    color = risk_color(prob)
    st.markdown(f"""
    <div class="risk-bar-container">
      <div class="risk-bar-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    <div style="display:flex; justify-content:space-between; margin-top:4px;">
      <span style="font-size:11px; color:#64748b;">Low</span>
      <span style="font-size:12px; color:{color}; font-weight:700;">{pct}%</span>
      <span style="font-size:11px; color:#64748b;">High</span>
    </div>
    """, unsafe_allow_html=True)

def render_result_card(label, css_cls, prob, model_name, extra=""):
    st.markdown(f"""
    <div class="{css_cls}" style="margin-top:16px;">
      <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:.08em; font-family:'JetBrains Mono',monospace;">
        {model_name}
      </div>
      <div style="font-size:26px; font-weight:700; color:#e2e8f0; margin:6px 0 2px;">
        {label}
      </div>
      <div style="font-size:13px; color:#94a3b8;">
        Confidence: <strong style="color:#e2e8f0;">{prob*100:.1f}%</strong>
        {"&nbsp;·&nbsp;" + extra if extra else ""}
      </div>
    </div>
    """, unsafe_allow_html=True)

def _get_ad_prob(model, proba_row):
    """
    Return (p_ad, p_neg, ad_label, neg_label) from a binary XGBoost/sklearn model.

    Works whether classes_ are integers or strings, and regardless of which
    index the model assigned to the positive (AD/Alzheimer's) class.

    Positive-class keywords  : ad, alzheimer, demented  (but NOT 'non' / 'control' / 'normal')
    Negative-class keywords  : control, normal, nondemented, healthy
    Fallback                 : index 1 = positive (sklearn convention)
    """
    # Get classes_ from pipeline's last step or directly
    try:
        classes = model.classes_
    except AttributeError:
        try:
            classes = model.steps[-1][1].classes_
        except Exception:
            classes = None

    AD_KEYS  = {"ad", "alzheimer", "alzheimers", "demented"}
    NEG_KEYS = {"control", "normal", "nondemented", "healthy", "cn"}

    ad_idx  = None
    neg_idx = None

    if classes is not None:
        for i, c in enumerate(classes):
            key = str(c).lower().replace(" ", "").replace("_", "").replace("'", "")
            # strip leading digits / underscores left by LabelEncoder
            key_stripped = key.lstrip("0123456789")
            if any(k in key for k in AD_KEYS) and not any(k in key for k in NEG_KEYS):
                ad_idx = i
            elif any(k in key for k in NEG_KEYS):
                neg_idx = i

    # Fallback: sklearn convention — class at index 1 is positive
    if ad_idx is None:
        ad_idx  = 1
    if neg_idx is None:
        neg_idx = 0 if ad_idx == 1 else 1

    # Resolve display labels from classes_ if strings, else use generic names
    def _label(idx, default):
        if classes is not None and isinstance(classes[idx], str):
            return classes[idx]
        return default

    p_ad  = float(proba_row[ad_idx])
    p_neg = float(proba_row[neg_idx])

    return p_ad, p_neg, _label(ad_idx, "AD"), _label(neg_idx, "Control")


def _two_class_note(p_ad, p_neg, ad_label, neg_label):
    """Render a compact 2-class probability bar for the details card."""
    col_ad  = "#ef4444" if p_ad >= 0.5 else "#64748b"
    col_neg = "#34d399" if p_neg >= 0.5 else "#64748b"
    return (
        f'<div style="margin-top:6px;">'
        # AD row
        f'<div style="font-size:10px;color:{col_ad};font-weight:{"600" if p_ad>=0.5 else "400"};margin:2px 0;">'
        f'{ad_label}{"  ◀" if p_ad>=0.5 else ""}</div>'
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
        f'<div style="flex:1;background:#0a1628;border-radius:4px;height:8px;overflow:hidden;">'
        f'<div style="width:{int(p_ad*100)}%;height:100%;background:{col_ad};border-radius:4px;"></div></div>'
        f'<span style="font-size:10px;color:{col_ad};font-weight:600;width:32px;">{p_ad*100:.0f}%</span></div>'
        # Negative row
        f'<div style="font-size:10px;color:{col_neg};font-weight:{"600" if p_neg>=0.5 else "400"};margin:2px 0;">'
        f'{neg_label}{"  ◀" if p_neg>=0.5 else ""}</div>'
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<div style="flex:1;background:#0a1628;border-radius:4px;height:8px;overflow:hidden;">'
        f'<div style="width:{int(p_neg*100)}%;height:100%;background:{col_neg};border-radius:4px;"></div></div>'
        f'<span style="font-size:10px;color:{col_neg};font-weight:600;width:32px;">{p_neg*100:.0f}%</span></div>'
        f'</div>'
    )



    ok = mdls.get(key) is not None
    cls = "dot-on" if ok else "dot-off"
    txt = "Loaded" if ok else "Missing"
    return f'<span class="{cls}"></span>{txt}'



# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def model_status_dot(mdls, key):
    """Return a coloured HTML dot indicating whether a model is loaded."""
    loaded = mdls.get(key) is not None
    color  = "#34d399" if loaded else "#ef4444"
    label  = "●"
    return (
        f'<span style="color:{color}; font-size:14px; line-height:1;">{label}</span>'
    )


def render_sidebar(mdls):
    with st.sidebar:
        st.markdown("""
        <div style="padding: 16px 0 20px;">
          <div style="font-size:20px; font-weight:700; color:#e2e8f0;">🧠 NeuroScan AI</div>
          <div style="font-size:11px; color:#475569; margin-top:2px;">Alzheimer's Risk Platform</div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🔬 Diagnosis"],
            key="nav",
            label_visibility="collapsed",
        )

        st.markdown("<div class='ns-divider'></div>", unsafe_allow_html=True)

        # Model status
        st.markdown("""
        <div style="font-size:10px; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
                    color:#475569; margin-bottom:10px; font-family:'JetBrains Mono',monospace;">
          Model Status
        </div>
        """, unsafe_allow_html=True)

        status_rows = [
            ("Clinical XGB",    "clin_model"),
            ("Biomarker XGB",   "bio_model"),
            ("Genetic Pipeline","gen_pipeline"),
            ("MRI DenseNet121", "mri_extractor"),
            ("MRI XGBoost",     "mri_xgb"),
        ]
        for label, key in status_rows:
            dot = model_status_dot(mdls, key)
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; font-size:12px;
                        color:#94a3b8; padding:4px 0;">
              <span>{label}</span>
              <span>{dot}</span>
            </div>
            """, unsafe_allow_html=True)

        # Show MRI error detail if it failed to load
        mri_err = mdls.get("mri_load_error", "")
        if mri_err:
            with st.expander("⚠ MRI load error", expanded=True):
                st.markdown("""
                <div style="font-size:12px; color:#fbbf24; font-weight:600; margin-bottom:8px;">
                  Quick fix — run ONE of these in your terminal, then click Reload Models:
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                            background:#050d1a; border-radius:6px; padding:8px 10px;
                            color:#34d399; margin-bottom:8px; line-height:1.8;">
                  pip install tf_keras<br>
                  <span style="color:#64748b;"># or if that doesn't work:</span><br>
                  pip install keras
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div style="font-size:10px; color:#475569; word-break:break-word;
                            line-height:1.5; font-family:'JetBrains Mono',monospace;
                            margin-top:6px;">
                  {mri_err}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='ns-divider'></div>", unsafe_allow_html=True)

        if st.button("🔄 Reload Models", use_container_width=True, key="reload_models"):
            st.cache_resource.clear()
            st.rerun()

    return page

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD PAGE
# ─────────────────────────────────────────────────────────────────────────────
def dashboard_page(mdls):
    st.markdown("""
    <div class="ns-card-accent" style="margin-bottom:24px;">
      <div class="ns-section-title">Overview</div>
      <div class="ns-title">NeuroScan AI</div>
      <div class="ns-subtitle">
        Integrates four independent machine learning models to provide
        a comprehensive Alzheimer's disease risk assessment from clinical scores,
        CSF biomarkers, gene expression, and MRI brain scans.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Model cards
    cols = st.columns(4)
    model_info = [
        ("🧪", "Clinical",   "XGBoost",            "14 cognitive & demographic features",           "clin_model"),
        ("🔬", "Biomarker",  "XGBoost",            "CSF Amyloid-β, p-Tau, t-Tau + demographics",   "bio_model"),
        ("🧬", "Genetic",    "SelectKBest+PCA+XGB", "Gene expression (GEO microarray data)",         "gen_pipeline"),
        ("🧠", "MRI Scan",   "DenseNet121+XGB",    "4-class dementia severity from brain MRI",      "mri_xgb"),
    ]
    for col, (icon, name, arch, desc, key) in zip(cols, model_info):
        loaded = mdls.get(key) is not None
        dot = "🟢" if loaded else "🔴"
        with col:
            st.markdown(f"""
            <div class="ns-card" style="height:180px; position:relative;">
              <div style="font-size:26px; margin-bottom:8px;">{icon}</div>
              <div style="font-size:14px; font-weight:700; color:#e2e8f0;">{name}</div>
              <div class="metric-pill" style="margin:6px 0;">{arch}</div>
              <div style="font-size:11px; color:#475569; margin-top:6px; line-height:1.45;">{desc}</div>
              <div style="position:absolute; top:16px; right:16px; font-size:11px;">{dot}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ns-info" style="margin-top:24px;">
      Head to <strong>Diagnosis</strong> in the sidebar to run a multi-modal analysis.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSIS PAGE
# ─────────────────────────────────────────────────────────────────────────────
def diagnosis_page(mdls):
    st.markdown("""
    <div class="ns-section-title">Analysis</div>
    <div class="ns-title" style="margin-bottom:4px;">Multi-Modal Diagnosis</div>
    <div class="ns-subtitle">
      Provide data for one or more modalities. Each available model will run independently,
      and a weighted ensemble risk score is computed.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧪 Clinical", "🔬 Biomarker", "🧬 Genetic", "🧠 MRI Scan"
    ])

    # ── Collect inputs ────────────────────────────────────────────────────────
    gen_file = mri_file = None

    # ═══ TAB 1 — CLINICAL ════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
        <div class="ns-card">
          <div class="ns-section-title">Clinical & Cognitive Features</div>
          <div class="ns-subtitle" style="margin-bottom:0;">
            Upload the clinical CSV file used during training. Required columns:
            <code>NACCAGE, SEX, EDUC, NACCAPOE, CDRGLOB, CDRSUM, NACCMOCA,
            TRAILA, TRAILB, ANIMALS, VEG, MEMUNITS, DIGIF, DIGIB</code>.
            Any missing columns will be imputed with population medians.
          </div>
        </div>
        """, unsafe_allow_html=True)

        loaded = mdls.get("clin_model") is not None
        if not loaded:
            st.markdown('<div class="ns-warning">⚠ Clinical model not loaded — place <code>clinical_xgb.pkl</code> and <code>clinical_scaler.pkl</code> in <code>models/</code></div>', unsafe_allow_html=True)

        clin_file = st.file_uploader(
            "Upload Clinical CSV",
            type=["csv"],
            key="clin_upload",
        )

        CLIN_FEATURES = [
            'NACCAGE', 'SEX', 'EDUC', 'NACCAPOE',
            'CDRGLOB', 'CDRSUM', 'NACCMOCA',
            'TRAILA', 'TRAILB', 'ANIMALS', 'VEG',
            'MEMUNITS', 'DIGIF', 'DIGIB',
        ]

        if clin_file:
            try:
                clin_df_raw = pd.read_csv(clin_file, low_memory=False)
                missing_cols = [c for c in CLIN_FEATURES if c not in clin_df_raw.columns]
                if missing_cols:
                    st.markdown(f'<div class="ns-warning">⚠ Missing columns: <code>{", ".join(missing_cols)}</code></div>', unsafe_allow_html=True)
                else:
                    clin_df = clin_df_raw[CLIN_FEATURES].copy()
                    clin_df = clin_df.fillna(clin_df.median(numeric_only=True))
                    st.session_state["clin_df"] = clin_df
                    st.success(f"✓ Loaded {len(clin_df):,} rows × {len(CLIN_FEATURES)} features from {clin_file.name}")
                    st.dataframe(clin_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")
        else:
            st.session_state.pop("clin_df", None)

    # ═══ TAB 2 — BIOMARKER ═══════════════════════════════════════════════════
    with tab2:
        st.markdown("""
        <div class="ns-card">
          <div class="ns-section-title">CSF Biomarker Panel</div>
          <div class="ns-subtitle" style="margin-bottom:0;">
            Upload the biomarker CSV file used during training. Required columns:
            <code>CSFABETA, CSFPTAU, CSFTTAU, NACCAGE, NACCSEX, EDUC, NACCAPOE</code>.
            These are the strongest biochemical predictors of Alzheimer's pathology.
          </div>
        </div>
        """, unsafe_allow_html=True)

        loaded_bio = mdls.get("bio_model") is not None
        if not loaded_bio:
            st.markdown('<div class="ns-warning">⚠ Biomarker model not loaded — place <code>biomarker_xgb_model.pkl</code> and <code>biomarker_scaler.pkl</code> in <code>models/</code></div>', unsafe_allow_html=True)

        bio_file = st.file_uploader(
            "Upload Biomarker CSV",
            type=["csv"],
            key="bio_upload",
        )

        BIO_FEATURES = ['CSFABETA', 'CSFPTAU', 'CSFTTAU', 'NACCAGE', 'NACCSEX', 'EDUC', 'NACCAPOE']

        if bio_file:
            try:
                bio_df_raw = pd.read_csv(bio_file, low_memory=False)
                missing_cols = [c for c in BIO_FEATURES if c not in bio_df_raw.columns]
                if missing_cols:
                    st.markdown(f'<div class="ns-warning">⚠ Missing columns: <code>{", ".join(missing_cols)}</code></div>', unsafe_allow_html=True)
                else:
                    bio_df = bio_df_raw[BIO_FEATURES].copy()
                    bio_df = bio_df.fillna(bio_df.median(numeric_only=True))
                    st.session_state["bio_df"] = bio_df
                    st.success(f"✓ Loaded {len(bio_df):,} rows × {len(BIO_FEATURES)} features from {bio_file.name}")
                    st.dataframe(bio_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")
        else:
            st.session_state.pop("bio_df", None)

    # ═══ TAB 3 — GENETIC ═════════════════════════════════════════════════════
    with tab3:
        st.markdown("""
        <div class="ns-card">
          <div class="ns-section-title">Gene Expression</div>
          <div class="ns-subtitle" style="margin-bottom:0;">
            Upload a GEO series matrix (.txt) or a pre-processed gene expression CSV.
            The pipeline (SelectKBest → StandardScaler → PCA → XGBoost) handles
            dimensionality reduction automatically.
          </div>
        </div>
        """, unsafe_allow_html=True)

        loaded_gen = mdls.get("gen_pipeline") is not None
        if not loaded_gen:
            st.markdown('<div class="ns-warning">⚠ Genetic model not loaded — place <code>gene_xgb_pipeline.pkl</code> in <code>models/</code></div>', unsafe_allow_html=True)

        gen_file = st.file_uploader(
            "Upload Gene Expression File (.txt GEO series matrix  or  .csv with gene columns)",
            type=["txt", "csv"],
            key="gen_upload",
        )
        if gen_file:
            st.session_state["gen_file_name"]  = gen_file.name
            st.session_state["gen_file_bytes"] = gen_file.read()
            st.success(f"✓ File received: {gen_file.name}")
        else:
            st.session_state.pop("gen_file_name",  None)
            st.session_state.pop("gen_file_bytes", None)

    # ═══ TAB 4 — MRI ═════════════════════════════════════════════════════════
    with tab4:
        st.markdown("""
        <div class="ns-card">
          <div class="ns-section-title">MRI Brain Scan</div>
          <div class="ns-subtitle" style="margin-bottom:0;">
            Upload an axial MRI slice (JPG / PNG). The DenseNet121 feature extractor
            feeds into XGBoost to classify dementia severity into 4 classes.
          </div>
        </div>
        """, unsafe_allow_html=True)

        mri_xgb_ok  = mdls.get("mri_xgb")       is not None
        mri_cnn_ok  = mdls.get("mri_extractor") is not None
        loaded_mri  = mri_xgb_ok and mri_cnn_ok
        mri_err     = mdls.get("mri_load_error", "")

        if not loaded_mri:
            xgb_icon = "✅" if mri_xgb_ok  else "❌"
            cnn_icon = "✅" if mri_cnn_ok  else "❌"
            tf_icon  = "✅" if TF_AVAILABLE else "❌"

            st.markdown(f"""
            <div class="ns-warning">
              <strong>⚠ MRI model not fully loaded.</strong>
              Check the checklist below and fix each ❌ item.
            </div>
            <div style="background:#050d1a; border:1px solid #1a2d45; border-radius:10px;
                        padding:16px 20px; margin-top:12px; font-size:13px; line-height:2;">
              <div>{tf_icon} &nbsp;<strong style="color:#94a3b8;">TensorFlow installed</strong>
                {"" if TF_AVAILABLE else f'<span style="color:#f87171; margin-left:8px;">— run: <code>pip install tensorflow</code></span>'}
              </div>
              <div>{xgb_icon} &nbsp;<strong style="color:#94a3b8;"><code>models/mri_xgb_densenet.pkl</code></strong>
                {"" if mri_xgb_ok else '<span style="color:#f87171; margin-left:8px;">— run the last cell of MRI-CNN.ipynb to generate this file</span>'}
              </div>
              <div>{cnn_icon} &nbsp;<strong style="color:#94a3b8;"><code>models/mri_densenet121_finetuned.keras</code></strong>
                {"" if mri_cnn_ok else '<span style="color:#f87171; margin-left:8px;">— run the last cell of MRI-CNN.ipynb to generate this file</span>'}
              </div>
              {f'<div style="margin-top:10px; color:#f87171; font-family:monospace; font-size:11px; word-break:break-all;"><strong>Error:</strong> {mri_err}</div>' if mri_err else ""}
            </div>
            """, unsafe_allow_html=True)

        mri_file = st.file_uploader("Upload MRI Image (JPG / PNG)", type=["jpg","jpeg","png"], key="mri_upload")

        if mri_file:
            col_img, col_info = st.columns([1, 2])
            img_bytes = mri_file.read()
            with col_img:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                st.image(img, caption="Uploaded MRI", use_container_width=True)
            with col_info:
                w, h = img.size
                st.markdown(f"""
                <div class="ns-metric" style="text-align:left; margin-bottom:10px;">
                  <div class="lbl">File</div>
                  <div style="font-size:14px; color:#e2e8f0; font-weight:600; margin-top:4px;">
                    {mri_file.name}
                  </div>
                </div>
                <div class="ns-metric" style="text-align:left;">
                  <div class="lbl">Original Resolution</div>
                  <div style="font-size:14px; color:#e2e8f0; font-weight:600; margin-top:4px;">
                    {w} × {h} px → resized to 224 × 224
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.session_state["mri_bytes"] = img_bytes
        else:
            st.session_state.pop("mri_bytes", None)

    # ═══ RUN FULL DIAGNOSIS ═══════════════════════════════════════════════════
    st.markdown("<div class='ns-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:12px;">
      <div class="ns-section-title">Ensemble Prediction</div>
      <div class="ns-subtitle" style="margin-bottom:0;">
        Click below to run all available models and compute a weighted risk score.
        At least one modality must be provided.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶  Run Full Diagnosis", key="run_diag", use_container_width=False):
        _run_diagnosis(mdls)


# ─────────────────────────────────────────────────────────────────────────────
# BINARY MODEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _find_ad_index(classes_, role="ad"):
    """Return the index that corresponds to the AD / Alzheimer's class.

    Works for integer labels (0/1) or string labels of any casing/spacing,
    e.g. 'AD', 'Alzheimer', 'alzheimers', 'dementia', '1'.
    Falls back to index 1 if nothing matches (historical default).
    """
    AD_KEYWORDS = {"ad", "alzheimer", "alzheimers", "alzheimer's",
                   "demented", "dementia", "mci", "1"}
    for idx, cls in enumerate(classes_):
        key = str(cls).lower().replace(" ", "").replace("_", "").replace("'", "")
        if key in AD_KEYWORDS or key.startswith("alz") or key.startswith("demen"):
            return idx
    # Integer fallback: if classes are [0, 1] assume 1 = AD
    try:
        return list(classes_).index(1)
    except ValueError:
        return 1   # last-resort default


def _two_class_html(p_ad, p_neg, pred_label, pos_name, neg_name, subtitle=""):
    """Return an HTML snippet with two probability bars for binary models."""
    pct_ad  = int(p_ad  * 100)
    pct_neg = int(p_neg * 100)
    col_ad  = "#ef4444"   # red for AD
    col_neg = "#34d399"   # green for Normal/Control
    pred_is_ad = pred_label.lower() not in {"normal", "control"}
    pred_color = col_ad if pred_is_ad else col_neg

    badge = (
        f'<span style="display:inline-block;padding:2px 9px;border-radius:99px;'
        f'font-size:10px;font-weight:700;background:{pred_color}22;'
        f'border:1px solid {pred_color}55;color:{pred_color};">'
        f'{pred_label}</span>'
    )
    sub = f'<div style="font-size:9px;color:#475569;margin-top:3px;">{subtitle}</div>' if subtitle else ""

    def bar(label, pct, color, is_pred):
        w = "font-weight:600;" if is_pred else ""
        arrow = " ◀" if is_pred else ""
        return (
            f'<div style="font-size:10px;color:#94a3b8;margin:4px 0 1px;{w}">{label}{arrow}</div>'
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="flex:1;background:#0a1628;border-radius:4px;height:8px;overflow:hidden;">'
            f'<div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div></div>'
            f'<span style="font-size:10px;color:{color};font-weight:600;width:32px;">{pct}%</span></div>'
        )

    return (
        f'{badge}{sub}'
        f'<div style="margin-top:7px;">'
        f'{bar(pos_name, pct_ad,  col_ad,  pred_is_ad)}'
        f'{bar(neg_name, pct_neg, col_neg, not pred_is_ad)}'
        f'</div>'
    )


def _run_diagnosis(mdls):
    probabilities, weights, modalities, details = [], [], [], []

    with st.spinner("Running models…"):
        time.sleep(0.3)  # brief pause so spinner is visible

        # ─── CLINICAL ─────────────────────────────────────────────────────
        clin_df = st.session_state.get("clin_df")
        if clin_df is not None and mdls.get("clin_model") and mdls.get("clin_scaler"):
            try:
                feats = [
                    'NACCAGE','SEX','EDUC','NACCAPOE',
                    'CDRGLOB','CDRSUM','NACCMOCA',
                    'TRAILA','TRAILB','ANIMALS','VEG',
                    'MEMUNITS','DIGIF','DIGIB'
                ]
                X        = clin_df[feats].values
                X_scaled = mdls["clin_scaler"].transform(X)
                proba_all   = mdls["clin_model"].predict_proba(X_scaled)
                raw_classes = mdls["clin_model"].classes_

                # Name-aware: find AD index regardless of label type / order
                ad_idx   = _find_ad_index(raw_classes, role="ad")
                ctrl_idx = 1 - ad_idx
                p_ad   = float(proba_all[:, ad_idx].mean())
                p_ctrl = float(proba_all[:, ctrl_idx].mean())
                pred_label = "AD" if p_ad >= 0.5 else "Control"

                probabilities.append(p_ad); weights.append(0.35); modalities.append("Clinical")
                details.append(("🧪 Clinical Model", p_ad,
                                 _two_class_html(p_ad, p_ctrl, pred_label,
                                                 "AD", "Control",
                                                 f"{len(clin_df):,} rows")))
            except Exception as e:
                st.warning(f"Clinical model error: {e}")

        # ─── BIOMARKER ────────────────────────────────────────────────────
        bio_df = st.session_state.get("bio_df")
        if bio_df is not None and mdls.get("bio_model") and mdls.get("bio_scaler"):
            try:
                feats       = ['CSFABETA','CSFPTAU','CSFTTAU','NACCAGE','NACCSEX','EDUC','NACCAPOE']
                X           = bio_df[feats].values
                X_scaled    = mdls["bio_scaler"].transform(X)
                proba_all   = mdls["bio_model"].predict_proba(X_scaled)
                raw_classes = mdls["bio_model"].classes_
                ad_idx      = _find_ad_index(raw_classes, role="ad")
                norm_idx    = 1 - ad_idx
                p_ad   = float(proba_all[:, ad_idx].mean())
                p_norm = float(proba_all[:, norm_idx].mean())
                pred_label = "AD" if p_ad >= 0.5 else "Normal"
                probabilities.append(p_ad); weights.append(0.25); modalities.append("Biomarker")
                details.append(("🔬 CSF Biomarker Model", p_ad,
                                 _two_class_html(p_ad, p_norm, pred_label,
                                                 "AD", "Normal",
                                                 f"{len(bio_df):,} rows")))
            except Exception as e:
                st.warning(f"Biomarker model error: {e}")

        # ─── GENETIC ──────────────────────────────────────────────────────
        gen_bytes = st.session_state.get("gen_file_bytes")
        gen_name  = st.session_state.get("gen_file_name", "")
        if gen_bytes and mdls.get("gen_pipeline"):
            try:
                if gen_name.endswith(".txt"):
                    df = pd.read_csv(io.BytesIO(gen_bytes), sep="\t", comment="!")
                    if "ID_REF" in df.columns:
                        df = df.set_index("ID_REF").T
                else:
                    df = pd.read_csv(io.BytesIO(gen_bytes))
                    for drop_col in ["Diagnosis","label","target"]:
                        if drop_col in df.columns:
                            df = df.drop(columns=[drop_col])

                gene_names = mdls.get("gen_genes")
                if gene_names:
                    for g in gene_names:
                        if g not in df.columns:
                            df[g] = 0.0
                    df = df[gene_names]

                df          = df.astype(float)
                proba_raw   = mdls["gen_pipeline"].predict_proba(df.head(1))[0]
                raw_classes = mdls["gen_pipeline"].classes_
                ad_idx      = _find_ad_index(raw_classes, role="ad")
                norm_idx    = 1 - ad_idx
                p_ad   = float(proba_raw[ad_idx])
                p_norm = float(proba_raw[norm_idx])
                pred_label = "Alzheimer's" if p_ad >= 0.5 else "Normal"
                probabilities.append(p_ad); weights.append(0.10); modalities.append("Genetic")
                details.append(("🧬 Gene Expression Model", p_ad,
                                 _two_class_html(p_ad, p_norm, pred_label,
                                                 "Alzheimer's", "Normal", "")))
            except Exception as e:
                st.warning(f"Genetic model error: {e}")

        # ─── MRI ──────────────────────────────────────────────────────────
        mri_bytes = st.session_state.get("mri_bytes")
        print("[v0] ═══════════════════════════════════════════════════════")
        print(f"[v0] MRI check: mri_bytes={mri_bytes is not None}, extractor={mdls.get('mri_extractor') is not None}, xgb={mdls.get('mri_xgb') is not None}")
        if mri_bytes and mdls.get("mri_extractor") and mdls.get("mri_xgb"):
            try:
                print("[v0] Starting MRI prediction pipeline...")
                img = Image.open(io.BytesIO(mri_bytes)).convert("RGB").resize((224, 224))
                arr = np.array(img, dtype=np.float32)
                arr = np.expand_dims(arr, 0)
                arr = preprocess_input(arr)
                feats = mdls["mri_extractor"].predict(arr, verbose=0)
                print(f"[v0] Features extracted: shape={feats.shape}")
                # Guard: extractor must output 1024-dim features (DenseNet121), not 4-class softmax
                if feats.shape[-1] != 1024:
                    raise RuntimeError(
                        f"Feature shape mismatch — extractor returned {feats.shape[-1]} features "
                        f"but XGBoost expects 1024. The model file may need to be re-saved, "
                        f"or click 'Reload Models' and try again."
                    )
                proba_raw = mdls["mri_xgb"].predict_proba(feats)[0]
                raw_classes = mdls["mri_xgb"].classes_
                print(f"[v0] Raw probabilities : {proba_raw}")
                print(f"[v0] XGBoost classes_  : {raw_classes}  (dtype={type(raw_classes[0]).__name__})")

                # ── Name-aware class alignment ────────────────────────────────
                # Works whether classes_ are strings (flow_from_directory) or
                # integers (LabelEncoder). Canonical key = lowercase, no spaces/underscores.
                CANONICAL = {
                    "nondemented":      ("Non Demented",       0.0),
                    "verymilddemented": ("Very Mild Demented", 0.3),
                    "milddemented":     ("Mild Demented",      0.6),
                    "moderatedemented": ("Moderate Demented",  1.0),
                }
                # Alphabetical int-index fallback (flow_from_directory default):
                #   0→MildDemented  1→ModerateDemented  2→NonDemented  3→VeryMildDemented
                ALPHA_ORDER = [
                    ("Mild Demented",      0.6),
                    ("Moderate Demented",  1.0),
                    ("Non Demented",       0.0),
                    ("Very Mild Demented", 0.3),
                ]

                class_prob_map = {}   # display_name → (probability, severity_weight)

                if isinstance(raw_classes[0], str):
                    # String labels — map by canonical name, robust to spacing/case
                    for cls, prob in zip(raw_classes, proba_raw):
                        key = cls.lower().replace(" ", "").replace("_", "")
                        if key in CANONICAL:
                            display, weight = CANONICAL[key]
                            class_prob_map[display] = (float(prob), weight)
                        else:
                            print(f"[v0] WARNING: Unrecognized class label '{cls}' — skipping")
                    if not class_prob_map:
                        raise RuntimeError(
                            f"No recognized class labels in XGBoost classes_: {list(raw_classes)}"
                        )
                else:
                    # Integer labels — sort ascending and apply alphabetical mapping
                    sorted_pairs = sorted(
                        zip([int(c) for c in raw_classes], proba_raw),
                        key=lambda x: x[0]
                    )
                    for i, (_, prob) in enumerate(sorted_pairs):
                        display, weight = ALPHA_ORDER[i]
                        class_prob_map[display] = (float(prob), weight)

                print("[v0] ───────────────────────────────────────────────────")
                print("[v0] CLASS PROBABILITIES (aligned):")
                for cls_name, (prob, weight) in class_prob_map.items():
                    print(f"[v0]   {cls_name:22s} | Prob: {prob:.6f} | SeverityWeight: {weight:.1f}")

                # Best class, weighted AD-risk score
                pred_class = max(class_prob_map, key=lambda k: class_prob_map[k][0])
                pred_conf  = class_prob_map[pred_class][0]
                p_ad       = float(sum(p * w for p, w in class_prob_map.values()))
                p_ad       = min(p_ad, 1.0)

                # ── Mild / Very Mild ambiguity detection ──────────────────────
                # These two adjacent classes are the hardest to separate.
                # If the runner-up is the other one and they're within 20 pp of
                # each other, bias toward Mild (clinically safer) and flag as
                # uncertain so the result is honest rather than falsely precise.
                AMBIGUOUS_PAIR   = {"Mild Demented", "Very Mild Demented"}
                p_mild      = class_prob_map.get("Mild Demented",      (0.0,))[0]
                p_verymild  = class_prob_map.get("Very Mild Demented",  (0.0,))[0]
                AMBIG_THRESH = 0.20   # within 20 percentage points → flag uncertain

                ambiguous = False
                if pred_class in AMBIGUOUS_PAIR:
                    gap = abs(p_mild - p_verymild)
                    if gap < AMBIG_THRESH:
                        ambiguous  = True
                        # Lean toward Mild: safer clinical call, and it's the
                        # class the model under-predicts vs the ground truth
                        pred_class = "Mild Demented"
                        pred_conf  = p_mild
                        print(f"[v0] Ambiguity detected (gap={gap:.3f}<{AMBIG_THRESH}) "
                              f"→ upgraded to Mild Demented")

                print("[v0] ───────────────────────────────────────────────────")
                print(f"[v0] PREDICTION : {pred_class}{'  [UNCERTAIN]' if ambiguous else ''}")
                print(f"[v0] Confidence : {pred_conf:.6f} ({pred_conf*100:.1f}%)")
                print(f"[v0] AD risk    : {p_ad:.6f}")
                print("[v0] ═══════════════════════════════════════════════════════")

                probabilities.append(p_ad); weights.append(0.40); modalities.append("MRI")

                # Uncertainty badge shown when Mild/VeryMild are close
                uncertain_badge = (
                    '<span style="display:inline-block;margin-left:6px;padding:1px 7px;'
                    'background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.35);'
                    'border-radius:99px;font-size:9px;color:#fbbf24;font-weight:600;'
                    'vertical-align:middle;">UNCERTAIN</span>'
                ) if ambiguous else ""

                # Build per-class bar string for the details card
                bar_lines = []
                severity_order = ["Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]
                bar_colors     = {"Non Demented":"#34d399","Very Mild Demented":"#60a5fa",
                                  "Mild Demented":"#f59e0b","Moderate Demented":"#ef4444"}
                for cls_name in severity_order:
                    if cls_name in class_prob_map:
                        p_cls, _ = class_prob_map[cls_name]
                        pct      = int(p_cls * 100)
                        col      = bar_colors.get(cls_name, "#94a3b8")
                        is_pred  = cls_name == pred_class
                        marker   = " ◀" if is_pred else ""
                        name_style = f"color:{col};font-weight:600;" if is_pred else ""
                        bar_lines.append(
                            f'<div style="margin:3px 0; font-size:10px; color:#94a3b8;{name_style}">'
                            f'{cls_name}{marker}</div>'
                            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">'
                            f'<div style="flex:1;background:#0a1628;border-radius:4px;height:8px;overflow:hidden;">'
                            f'<div style="width:{pct}%;height:100%;background:{col};border-radius:4px;"></div></div>'
                            f'<span style="font-size:10px;color:{col};font-weight:600;width:32px;">{pct}%</span></div>'
                        )
                class_breakdown_html = "".join(bar_lines)

                ambig_note = (
                    '<div style="margin-top:6px;padding:5px 8px;background:rgba(245,158,11,0.08);'
                    'border:1px solid rgba(245,158,11,0.2);border-radius:6px;font-size:9px;'
                    'color:#fbbf24;line-height:1.4;">'
                    '⚠ Mild &amp; Very Mild probabilities are close — '
                    'result biased toward Mild (clinically safer).'
                    '</div>'
                ) if ambiguous else ""

                details.append(("🧠 MRI Model", p_ad,
                                 f'Class: <strong>{pred_class}</strong> '
                                 f'({pred_conf*100:.1f}%){uncertain_badge}<br>'
                                 f'<div style="margin-top:6px;">{class_breakdown_html}</div>'
                                 f'{ambig_note}'))

            except Exception as e:
                print(f"[v0] MRI MODEL ERROR: {e}")
                import traceback
                print(f"[v0] Traceback: {traceback.format_exc()}")
                st.warning(f"MRI model error: {e}")
        else:
            print("[v0] MRI pipeline skipped (missing components)")

    # ═══ RESULTS ═════════════════════════════════════════════════════════════
    if not probabilities:
        st.markdown('<div class="ns-warning">⚠ No modality data detected. Submit at least one form or upload a file first.</div>', unsafe_allow_html=True)
        return

    final = float(np.average(probabilities, weights=weights))
    rlabel, rcls = risk_label(final)
    rcolor = risk_color(final)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Main risk card
    st.markdown(f"""
    <div class="{rcls}">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
        <div>
          <div style="font-size:11px; color:#94a3b8; text-transform:uppercase;
                      letter-spacing:.1em; font-family:'JetBrains Mono',monospace; margin-bottom:6px;">
            Ensemble Alzheimer's Risk Score
          </div>
          <div style="font-size:48px; font-weight:800; color:{rcolor}; line-height:1;">
            {final*100:.1f}%
          </div>
          <div style="font-size:16px; font-weight:600; color:{rcolor}; margin-top:4px;">
            {rlabel}
          </div>
        </div>
        <div style="text-align:right;">
          <div class="metric-pill">
            {len(modalities)} modali{'ty' if len(modalities)==1 else 'ties'}
          </div>
          <div style="font-size:11px; color:#64748b; margin-top:8px;">
            {" · ".join(modalities)}
          </div>
        </div>
      </div>
      <div style="margin-top:20px;">
    """, unsafe_allow_html=True)
    render_risk_bar(final)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Per-model breakdown
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="ns-section-title" style="margin-bottom:12px;">Per-Model Breakdown</div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(details))
    for col, (mname, prob, note) in zip(cols, details):
        color = risk_color(prob)
        with col:
            st.markdown(f"""
            <div class="ns-metric">
              <div class="val" style="color:{color};">{prob*100:.0f}%</div>
              <div class="lbl">{mname}</div>
              <div style="font-size:10px; color:#475569; margin-top:6px; line-height:1.3;">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div style="margin-top:20px; padding:12px 16px; background:#050d1a; border-radius:8px;
                font-size:11px; color:#475569; line-height:1.6; border:1px solid #0d1f35;">
      ⚠ <strong style="color:#64748b;">Medical Disclaimer:</strong> This tool is for
      research and educational purposes only. Results do not constitute medical advice,
      diagnosis, or treatment. Always consult a qualified neurologist for clinical decisions.
    </div>
    """, unsafe_allow_html=True)

    # Clear cached inputs so next run is fresh
    for k in ["clin_df", "bio_df", "gen_file_bytes", "gen_file_name", "mri_bytes"]:
        st.session_state.pop(k, None)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    mdls = load_models()
    page = render_sidebar(mdls)

    if page == "🏠 Dashboard":
        dashboard_page(mdls)
    elif page == "🔬 Diagnosis":
        diagnosis_page(mdls)


if __name__ == "__main__":
    main()