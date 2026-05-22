"""
RetinAI — Systeme de Diagnostic de Retinopathie Diabetique
Ensemble de 4 CNNs (ResNet50 · EfficientNet-B0 · DenseNet121 · ResNet18)
Grad-CAM · TTA × 4 vues · Rapport PDF telechargeable
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import os
import tempfile
from datetime import datetime
from pathlib import Path

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetinAI — Diagnostic RD",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constantes ───────────────────────────────────────────────────────────────
CLASSES     = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
CLASSES_FR  = ["Absence de RD", "Legere", "Moderee", "Severe", "Proliferative"]  # sans accents pour PDF
CLASSES_UI  = ["Absence de RD", "Légère", "Modérée", "Sévère", "Proliférative"]  # avec accents pour UI
MODEL_DIR   = Path(__file__).parent / "models"
MODEL_NAMES = ["resnet50", "efficientnet_b0", "densenet121", "resnet18"]
IMG_SIZE    = 224
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

VAL_ACCS = {
    "resnet50": 0.992, "efficientnet_b0": 0.992,
    "densenet121": 0.990, "resnet18": 0.990,
}

SEV = {
    0: {"color": "#27ae60", "bg": "#eafaf1", "urgency": "Aucune",   "urgency_pdf": "Aucune",
        "rec": "Controle annuel recommande. Maintenir le controle glycemique et tensionnel.",
        "rec_ui": "Contrôle annuel recommandé. Maintenir le contrôle glycémique et tensionnel."},
    1: {"color": "#f39c12", "bg": "#fef9e7", "urgency": "Faible",   "urgency_pdf": "Faible",
        "rec": "Suivi ophtalmologique dans les 12 mois. Optimiser la glycemie et la tension arterielle.",
        "rec_ui": "Suivi ophtalmologique dans les 12 mois. Optimiser la glycémie et la tension artérielle."},
    2: {"color": "#e67e22", "bg": "#fdf2e9", "urgency": "Modérée",  "urgency_pdf": "Moderee",
        "rec": "Consultation specialiste retine dans les 3 a 6 mois. Traitement medicamenteux a envisager.",
        "rec_ui": "Consultation spécialiste rétine dans les 3 à 6 mois. Traitement médicamenteux à envisager."},
    3: {"color": "#e74c3c", "bg": "#fdedec", "urgency": "Élevée",   "urgency_pdf": "Elevee",
        "rec": "Reference urgente au specialiste retine sous 1 mois. Envisager anti-VEGF ou photocoagulation laser.",
        "rec_ui": "Référence urgente au spécialiste rétine sous 1 mois. Envisager anti-VEGF ou photocoagulation laser."},
    4: {"color": "#8e44ad", "bg": "#f5eef8", "urgency": "Critique", "urgency_pdf": "Critique",
        "rec": "Traitement immediat requis - reference en urgence. Risque de cecite imminente.",
        "rec_ui": "Traitement immédiat requis — référence en urgence. Risque de cécité imminente."},
}

EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ─── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "pred": None, "ensemble_probs": None, "per_model_probs": None,
    "analysis_done": False, "last_file_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
[data-testid="stSidebar"] * { color: #dde1f0 !important; }
[data-testid="stSidebar"] hr { border-color: #2a2a5e !important; }

.main-title {
    font-size: 2.2rem; font-weight: 900; color: #1a1a2e;
    letter-spacing: -0.5px; margin-bottom: 0;
}
.sub-title {
    font-size: 0.9rem; color: #7f8c8d;
    margin-top: 0.2rem; margin-bottom: 1.5rem;
}
.section-header {
    font-size: 1rem; font-weight: 700; color: #1a1a2e;
    margin: 1.4rem 0 0.6rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #e8ecef;
}
.diag-card {
    border-radius: 14px; padding: 1.4rem 1.8rem;
    margin: 0.8rem 0; box-shadow: 0 4px 20px rgba(0,0,0,0.10);
}
.urgency-badge {
    display: inline-block; padding: 0.2rem 0.7rem;
    border-radius: 20px; font-size: 0.76rem;
    font-weight: 700; color: white; margin-left: 0.6rem;
    vertical-align: middle;
}
.disclaimer {
    background: #fff8e7; border: 1px solid #f39c12;
    border-radius: 8px; padding: 0.8rem 1rem;
    font-size: 0.82rem; color: #7d6608; margin-top: 1rem;
}
.upload-hint {
    border: 2px dashed #3498db; border-radius: 12px;
    padding: 2.5rem 2rem; text-align: center; background: #f8fbff;
}
.demo-banner {
    background: #fff3cd; border: 2px solid #ffc107;
    border-radius: 10px; padding: 0.8rem 1.2rem;
    font-size: 0.88rem; color: #856404; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ─── Fonctions métier ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def build_model(name: str) -> nn.Module:
    if name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 5)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 5)
    elif name == "densenet121":
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, 5)
    elif name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 5)
    else:
        raise ValueError(f"Modele inconnu : {name}")
    return m


@st.cache_resource(show_spinner="Chargement des modeles CNN…")
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded, missing = {}, []
    for name in MODEL_NAMES:
        path = MODEL_DIR / f"{name}_advanced.pth"
        if path.exists():
            try:
                m = build_model(name)
                m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                m.to(device).eval()
                loaded[name] = m
            except Exception as e:
                st.warning(f"Erreur chargement {name} : {e}")
                missing.append(name)
        else:
            missing.append(name)
    return loaded, missing, device


def demo_predict(img_pil: Image.Image) -> tuple:
    """Prédiction de démonstration (sans vrais modèles)."""
    np.random.seed(int(np.array(img_pil.resize((8, 8))).mean()))
    raw = np.random.dirichlet(alpha=[5, 1, 2, 0.5, 0.5])
    pred = int(raw.argmax())
    per_model = {n: np.random.dirichlet(alpha=[5, 1, 2, 0.5, 0.5]) for n in MODEL_NAMES}
    return pred, raw, per_model


@torch.no_grad()
def predict(img_pil: Image.Image, loaded_models: dict, device: torch.device) -> tuple:
    x = EVAL_TF(img_pil).unsqueeze(0).to(device)
    per_model: dict[str, np.ndarray] = {}

    for name, model in loaded_models.items():
        views = [
            x,
            torch.flip(x, dims=[3]),
            torch.flip(x, dims=[2]),
            torch.rot90(x, k=2, dims=[2, 3]),
        ]
        avg = torch.stack([F.softmax(model(v), dim=1) for v in views]).mean(0)
        per_model[name] = avg.cpu().numpy()[0]

    names = list(per_model.keys())
    w = np.array([VAL_ACCS.get(n, 1.0) for n in names], dtype=np.float32)
    w /= w.sum()
    ensemble = np.sum([w[i] * per_model[n] for i, n in enumerate(names)], axis=0)
    pred = int(ensemble.argmax())
    return pred, ensemble, per_model


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.acts = self.grads = None
        self._fh = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "acts", o.detach().clone()))
        self._bh = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grads", go[0].detach().clone()))

    def __call__(self, x: torch.Tensor, cls: int) -> np.ndarray:
        self.model.eval()
        logits = self.model(x)
        self.model.zero_grad()
        logits[0, cls].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def remove(self):
        self._fh.remove()
        self._bh.remove()


def get_target_layer(model: nn.Module, name: str) -> nn.Module:
    if name in ("resnet50", "resnet18"):  return model.layer4[-1]
    if name == "densenet121":             return model.features.denseblock4  # pas features[-1] → inplace relu bug
    if name == "efficientnet_b0":         return model.features[-1]
    raise ValueError(name)


def compute_gradcam_fig(
    img_pil: Image.Image, model: nn.Module,
    model_name: str, device: torch.device, pred_class: int
) -> io.BytesIO:
    img_t = EVAL_TF(img_pil).unsqueeze(0).to(device)
    with torch.enable_grad():
        cam_engine = GradCAM(model, get_target_layer(model, model_name))
        cam = cam_engine(img_t, pred_class)
        cam_engine.remove()

    inv_mean = torch.tensor(MEAN).view(3, 1, 1)
    inv_std  = torch.tensor(STD).view(3, 1, 1)
    orig = (img_t.squeeze().cpu() * inv_std + inv_mean).clamp(0, 1).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#f8f9fa")
    for ax in axes:
        ax.set_facecolor("#f8f9fa")

    axes[0].imshow(orig)
    axes[0].set_title("Image originale", fontsize=12, fontweight="bold", color="#1a1a2e")
    axes[0].axis("off")

    axes[1].imshow(orig)
    axes[1].imshow(cam, cmap="jet", alpha=0.45)
    axes[1].set_title(f"Zones d'attention — {model_name}", fontsize=12,
                      fontweight="bold", color="#1a1a2e")
    axes[1].axis("off")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.035, pad=0.02)
    cbar.set_label("Activation", fontsize=8, color="#555")
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout(pad=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def demo_gradcam_fig(img_pil: Image.Image) -> io.BytesIO:
    """Grad-CAM factice pour le mode démo."""
    arr = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(float) / 255.0
    cam = arr.mean(axis=2)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#f8f9fa")
    axes[0].imshow(arr); axes[0].set_title("Image originale", fontweight="bold"); axes[0].axis("off")
    axes[1].imshow(arr); axes[1].imshow(cam, cmap="jet", alpha=0.45)
    axes[1].set_title("Grad-CAM (DEMO)", fontweight="bold"); axes[1].axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ─── Graphique de confiance ───────────────────────────────────────────────────
def plot_confidence(ensemble_probs: np.ndarray, pred: int) -> io.BytesIO:
    colors = ["#27ae60", "#f39c12", "#e67e22", "#e74c3c", "#8e44ad"]
    edge   = ["#1a8a4a", "#c07d0e", "#b56218", "#b03030", "#6c3483"]
    probs_pct = ensemble_probs * 100

    fig, ax = plt.subplots(figsize=(8, 3.4))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    bars = ax.barh(
        CLASSES_UI[::-1], probs_pct[::-1],
        color=colors[::-1], edgecolor=edge[::-1],
        linewidth=[2.5 if (4 - i) == pred else 0.5 for i in range(5)],
        height=0.6,
    )
    for bar, val, i_rev in zip(bars, probs_pct[::-1], range(5)):
        bold = (4 - i_rev) == pred
        ax.text(
            min(val + 0.6, 102), bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", ha="left",
            fontsize=10.5 if bold else 9.5,
            fontweight="bold" if bold else "normal",
            color="#1a1a2e",
        )

    ax.set_xlim(0, 112)
    ax.set_xlabel("Confiance (%)", fontsize=10, color="#555")
    ax.set_title(
        "Distribution des probabilites — Ensemble pondere (4 modeles + TTA x4)",
        fontsize=10, fontweight="bold", color="#1a1a2e", pad=8
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10.5, colors="#333")
    ax.tick_params(axis="x", colors="#888")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ─── Génération PDF ───────────────────────────────────────────────────────────
def generate_pdf(
    patient_name: str, patient_age: str, doctor_name: str,
    pred: int, ensemble_probs: np.ndarray,
    img_pil: Image.Image,
    chart_buf: io.BytesIO,
    cam_buf: io.BytesIO,
    is_demo: bool = False,
) -> bytes | None:
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    sev = SEV[pred]
    r, g, b = tuple(int(sev["color"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── En-tete ──────────────────────────────────────────────────
    pdf.set_fill_color(26, 26, 46)
    pdf.rect(0, 0, 210, 33, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(10, 7)
    pdf.cell(0, 10, "RetinAI  -  Rapport de Diagnostic", ln=1)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_xy(10, 22)
    pdf.cell(
        0, 6,
        "Systeme d'aide au diagnostic de Retinopathie Diabetique  |  "
        f"Genere le {datetime.now().strftime('%d/%m/%Y a %H:%M')}"
        + ("  |  [MODE DEMO]" if is_demo else "")
    )

    if is_demo:
        pdf.set_fill_color(255, 193, 7)
        pdf.set_text_color(80, 60, 0)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_xy(0, 33)
        pdf.cell(210, 7, "  ATTENTION : Resultats generes en mode demonstration — aucun modele reel charge", fill=True, ln=1)
        pdf.set_text_color(0, 0, 0)
        pdf.set_y(44)
    else:
        pdf.set_text_color(0, 0, 0)
        pdf.set_y(41)

    # ── Informations patient ──────────────────────────────────────
    pdf.set_fill_color(235, 240, 248)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "  Informations Patient", ln=1, fill=True)
    pdf.ln(2)
    rows = [
        ("Patient",       patient_name or "Non renseigne"),
        ("Age",           f"{patient_age} ans" if patient_age else "Non renseigne"),
        ("Medecin",       doctor_name or "Non renseigne"),
        ("Date analyse",  datetime.now().strftime("%d/%m/%Y a %H:%M")),
        ("Methode",       "Ensemble 4 CNNs + TTA x4 vues" + (" (DEMO)" if is_demo else "")),
    ]
    for label, val in rows:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(48, 6, f"    {label} :")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, str(val), ln=1)
    pdf.ln(4)

    # ── Bandeau diagnostic ────────────────────────────────────────
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(
        0, 11,
        f"  Diagnostic : {CLASSES_FR[pred].upper()}  (Grade {pred}/4)"
        f"  |  Urgence : {sev['urgency_pdf']}",
        ln=1, fill=True
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, "  Recommandation clinique :", ln=1)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_x(14)
    pdf.multi_cell(0, 5, sev["rec"])
    pdf.ln(5)

    # ── Tableau des scores ────────────────────────────────────────
    pdf.set_fill_color(235, 240, 248)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "  Scores de confiance (ensemble 4 modeles + TTA)", ln=1, fill=True)
    pdf.ln(2)

    pdf.set_fill_color(210, 220, 235)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(80, 7, "  Grade (FR)", border=1, fill=True)
    pdf.cell(55, 7, "  Grade (EN)", border=1, fill=True)
    pdf.cell(35, 7, "  Confiance", border=1, fill=True)
    pdf.cell(25, 7, "  Statut", border=1, fill=True, ln=1)

    pdf.set_font("Helvetica", "", 9)
    for i, (cls_fr, cls_en, prob) in enumerate(zip(CLASSES_FR, CLASSES, ensemble_probs)):
        is_pred = (i == pred)
        if is_pred:
            pdf.set_fill_color(r, g, b)
            pdf.set_text_color(255, 255, 255)
        else:
            pdf.set_fill_color(248, 249, 250)
        pdf.cell(80, 6, f"  Grade {i} - {cls_fr}", border=1, fill=True)
        pdf.cell(55, 6, f"  {cls_en}", border=1, fill=True)
        pdf.cell(35, 6, f"  {prob*100:.1f}%", border=1, fill=True)
        pdf.cell(25, 6, "  OK" if is_pred else "", border=1, fill=True, ln=1)
        if is_pred:
            pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # ── Images : ecriture dans des fichiers temporaires ───────────
    tmp_files = []

    def save_tmp(content_or_pil, suffix: str) -> str:
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        if isinstance(content_or_pil, bytes):
            f.write(content_or_pil)
        else:
            content_or_pil.convert("RGB").resize((400, 400)).save(f.name, "JPEG", quality=92)
            f.close()
            return f.name
        f.close()
        return f.name

    img_path   = save_tmp(img_pil, ".jpg")
    chart_path = save_tmp(chart_buf.getvalue(), ".png")
    cam_path   = save_tmp(cam_buf.getvalue(), ".png")
    tmp_files  = [img_path, chart_path, cam_path]

    # Image retinienne
    pdf.set_fill_color(235, 240, 248)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "  Image de fond d'oeil analysee", ln=1, fill=True)
    pdf.ln(2)
    pdf.image(img_path, x=55, w=100)
    pdf.ln(4)

    # Graphique probabilites
    pdf.set_fill_color(235, 240, 248)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "  Distribution des probabilites par classe", ln=1, fill=True)
    pdf.ln(2)
    pdf.image(chart_path, x=8, w=194)
    pdf.ln(4)

    # Grad-CAM
    pdf.set_fill_color(235, 240, 248)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "  Interpretabilite - Grad-CAM (zones d'attention du reseau)", ln=1, fill=True)
    pdf.ln(2)
    pdf.image(cam_path, x=8, w=194)
    pdf.ln(4)

    # Avertissement
    pdf.set_fill_color(255, 248, 220)
    pdf.set_text_color(120, 80, 10)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.multi_cell(
        0, 5,
        "AVERTISSEMENT : Ce rapport est produit par un systeme d'intelligence artificielle "
        "et constitue une aide au diagnostic uniquement. Il ne se substitue pas a l'examen "
        "clinique ni a l'avis d'un ophtalmologiste qualifie. Toute decision therapeutique "
        "doit etre validee par un professionnel de sante habilite.",
        fill=True,
    )

    for p in tmp_files:
        try:
            os.unlink(p)
        except OSError:
            pass

    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════════════════════
# ─── INTERFACE ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── Chargement des modèles ────────────────────────────────────────────────────
loaded_models, missing_models, device = load_models()
demo_mode = len(loaded_models) == 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👁️ RetinAI")
    st.markdown("**Diagnostic Rétinopathie Diabétique**")
    st.markdown("---")

    # Statut modèles
    if demo_mode:
        st.warning("⚠️ Mode démonstration\nAucun modèle chargé.")
    else:
        st.success(f"✅ {len(loaded_models)}/4 modèles chargés\n`{device}`")

    st.markdown("---")
    st.markdown("### Modèles CNN")
    for name in MODEL_NAMES:
        p = MODEL_DIR / f"{name}_advanced.pth"
        icon = "✅" if p.exists() else "❌"
        acc  = f"{VAL_ACCS[name]*100:.1f}%" if p.exists() else "manquant"
        st.markdown(f"{icon} **{name}** — {acc}")

    st.markdown("---")
    st.markdown("### Paramètres")
    show_gradcam    = st.toggle("Afficher Grad-CAM", value=True)
    cam_model_choice = st.selectbox(
        "Modèle Grad-CAM",
        options=[n for n in MODEL_NAMES if (MODEL_DIR / f"{n}_advanced.pth").exists()]
                 or MODEL_NAMES,
        index=0,
        help="Architecture utilisée pour la carte de chaleur Grad-CAM"
    )

    st.markdown("---")
    st.markdown("### Performances (notebook 06)")
    st.info(
        "Test set équilibré (250 images)\n\n"
        "**Accuracy :** 99.6 %\n\n"
        "**MAGE :** 0.016\n\n"
        "**Within-1-grade :** 99.6 %"
    )

    if not demo_mode:
        st.markdown("---")
        st.caption(
            "Pour obtenir les poids :\n"
            "Exécuter `06_final_dataset2_best_results_996pct.ipynb` "
            "sur Kaggle → onglet *Output* → télécharger les `.pth`"
        )

# ── En-tête principal ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">👁️ RetinAI — Diagnostic Rétinopathie Diabétique</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">'
    "Ensemble 4 CNNs &nbsp;·&nbsp; Test-Time Augmentation ×4 &nbsp;·&nbsp; "
    "Grad-CAM &nbsp;·&nbsp; Rapport PDF &nbsp;·&nbsp; 2× Tesla T4"
    "</div>",
    unsafe_allow_html=True,
)

if demo_mode:
    st.markdown("""
    <div class="demo-banner">
    🎓 <strong>Mode démonstration actif</strong> — Les fichiers <code>.pth</code> ne sont pas
    présents dans <code>models/</code>. Les prédictions affichées sont <em>simulées</em>
    pour permettre de visualiser l'interface. Placez les vrais poids pour activer l'analyse réelle.
    </div>
    """, unsafe_allow_html=True)

# ── Informations patient ──────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">📋 Informations Patient (optionnel)</div>',
    unsafe_allow_html=True,
)
c1, c2, c3 = st.columns(3)
with c1:
    patient_name = st.text_input("Nom du patient", placeholder="Ex : Jean Dupont")
with c2:
    patient_age = st.text_input("Âge", placeholder="Ex : 58")
with c3:
    doctor_name = st.text_input("Médecin référent", placeholder="Dr. ...")

# ── Upload image ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">🖼️ Image de fond d\'œil</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Glissez-deposez ou cliquez pour charger une image de retine",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if not uploaded:
    st.markdown("""
    <div class="upload-hint">
        <div style="font-size:3rem">🩺</div>
        <h4 style="color:#3498db;margin:0.5rem 0 0.3rem">
            Glissez-déposez une image fundus ici
        </h4>
        <p style="margin:0;color:#999;font-size:0.9rem">
            Formats acceptés : PNG &nbsp;·&nbsp; JPG &nbsp;·&nbsp; JPEG
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Réinitialiser si nouvelle image
file_id = uploaded.file_id if hasattr(uploaded, "file_id") else uploaded.name
if file_id != st.session_state.last_file_id:
    st.session_state.analysis_done = False
    st.session_state.pred = None
    st.session_state.last_file_id = file_id

img_pil = Image.open(uploaded).convert("RGB")

# ── Aperçu image ──────────────────────────────────────────────────────────────
col_img, col_meta = st.columns([1, 2])
with col_img:
    st.image(img_pil, caption="Image chargée", use_container_width=True)
with col_meta:
    st.markdown("**Détails de l'image**")
    st.write(f"- **Dimensions :** {img_pil.size[0]} × {img_pil.size[1]} px")
    st.write(f"- **Format :** {uploaded.type}")
    st.write(f"- **Taille fichier :** {uploaded.size / 1024:.1f} Ko")
    if demo_mode:
        st.write("- **Modèle :** ⚠️ Mode démonstration")
    else:
        st.write(f"- **Modèles actifs :** {', '.join(loaded_models.keys())}")
    st.caption(
        "L'image sera redimensionnée en 224×224 px. "
        "TTA appliqué sur 4 vues : original, flip H, flip V, rotation 180°."
    )

# ── Bouton Analyser ───────────────────────────────────────────────────────────
st.markdown("---")
col_btn, col_reset = st.columns([4, 1])
with col_btn:
    run = st.button(
        "🔍 Lancer l'analyse" + (" (DEMO)" if demo_mode else ""),
        type="primary", use_container_width=True,
    )
with col_reset:
    if st.button("🔄 Reset", use_container_width=True) and st.session_state.analysis_done:
        st.session_state.analysis_done = False
        st.rerun()

# ── Lancement de l'analyse ────────────────────────────────────────────────────
if run:
    with st.spinner("Inférence en cours…"):
        if demo_mode:
            pred, ensemble_probs, per_model_probs = demo_predict(img_pil)
        else:
            pred, ensemble_probs, per_model_probs = predict(img_pil, loaded_models, device)

    st.session_state.pred            = pred
    st.session_state.ensemble_probs  = ensemble_probs
    st.session_state.per_model_probs = per_model_probs
    st.session_state.analysis_done   = True

# ── Affichage des résultats ───────────────────────────────────────────────────
if st.session_state.analysis_done:
    pred           = st.session_state.pred
    ensemble_probs = st.session_state.ensemble_probs
    per_model_probs = st.session_state.per_model_probs
    sev = SEV[pred]

    urg_colors = {
        "Aucune": "#27ae60", "Faible": "#f39c12",
        "Modérée": "#e67e22", "Élevée": "#e74c3c", "Critique": "#8e44ad",
    }
    urg_col = urg_colors[sev["urgency"]]

    # ── Carte de diagnostic ───────────────────────────────────────
    st.markdown("### 📊 Résultats du diagnostic")
    st.markdown(f"""
    <div class="diag-card"
         style="background:{sev['bg']}; border-left:6px solid {sev['color']};">
        <h2 style="color:{sev['color']};margin:0 0 0.4rem 0;font-size:1.85rem;">
            Grade {pred} &nbsp;—&nbsp; {CLASSES_UI[pred]}
            <span class="urgency-badge" style="background:{urg_col};">
                Urgence : {sev['urgency']}
            </span>
            {"<span class='urgency-badge' style='background:#999;'>DEMO</span>" if demo_mode else ""}
        </h2>
        <p style="color:#555;margin:0;font-size:0.95rem;line-height:1.5">
            🩺 <strong>Recommandation clinique :</strong> {sev['rec_ui']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Métriques ─────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Grade prédit",        f"{pred} / 4")
    m2.metric("Confiance ensemble",  f"{ensemble_probs[pred]*100:.1f} %")
    m3.metric("Modèles en accord",   len(per_model_probs))
    m4.metric("Vues TTA",            4)

    # ── Graphique de confiance ────────────────────────────────────
    st.markdown(
        '<div class="section-header">📈 Distribution des probabilités</div>',
        unsafe_allow_html=True,
    )
    chart_buf = plot_confidence(ensemble_probs, pred)
    st.image(chart_buf, use_container_width=True)

    # ── Vote détaillé par modèle ──────────────────────────────────
    with st.expander("🔬 Vote détaillé par modèle (TTA inclus)"):
        detail_cols = st.columns(len(per_model_probs))
        for col, (name, probs) in zip(detail_cols, per_model_probs.items()):
            mp = int(probs.argmax())
            with col:
                st.markdown(f"**{name}**")
                st.caption(f"Grade {mp} · {CLASSES_UI[mp]}")
                st.caption(f"Confiance : **{probs[mp]*100:.1f}%**")
                for i, (cls_ui, p) in enumerate(zip(CLASSES_UI, probs)):
                    st.progress(float(p), text=f"G{i}: {p*100:.0f}%")

    # ── Grad-CAM ─────────────────────────────────────────────────
    if show_gradcam:
        st.markdown(
            '<div class="section-header">🔬 Interprétabilité — Grad-CAM</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Les zones **rouges / chaudes** indiquent les régions rétiniennes qui ont "
            "le plus influencé la décision du réseau. Un modèle fiable active sur les "
            "**lésions** (micro-anévrismes, hémorragies, exsudats) et non sur le fond noir."
        )
        with st.spinner(f"Calcul Grad-CAM ({cam_model_choice})…"):
            if demo_mode:
                cam_buf = demo_gradcam_fig(img_pil)
            else:
                cam_m = cam_model_choice if cam_model_choice in loaded_models \
                        else next(iter(loaded_models))
                cam_buf = compute_gradcam_fig(img_pil, loaded_models[cam_m], cam_m, device, pred)

        st.image(cam_buf, use_container_width=True)
    else:
        # Toujours calculer pour le PDF, même si non affiché
        if demo_mode:
            cam_buf = demo_gradcam_fig(img_pil)
        else:
            cam_m = cam_model_choice if cam_model_choice in loaded_models \
                    else next(iter(loaded_models))
            cam_buf = compute_gradcam_fig(img_pil, loaded_models[cam_m], cam_m, device, pred)

    # ── Rapport PDF ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">📄 Rapport PDF du diagnostic</div>',
        unsafe_allow_html=True,
    )

    chart_buf.seek(0)
    cam_buf.seek(0)

    try:
        pdf_bytes = generate_pdf(
            patient_name, patient_age, doctor_name,
            pred, ensemble_probs,
            img_pil, chart_buf, cam_buf,
            is_demo=demo_mode,
        )
        if pdf_bytes:
            safe = (patient_name or "patient").replace(" ", "_")
            fname = f"RetinAI_{safe}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                label="⬇️  Télécharger le rapport PDF",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
            st.caption(
                "Le rapport inclut : informations patient · diagnostic avec urgence · "
                "scores de confiance par grade · image rétinienne · graphique de probabilités · "
                "carte Grad-CAM · recommandation clinique · avertissement médical."
            )
        else:
            st.info("Installez `fpdf2` pour la génération PDF : `pip install fpdf2`")

    except Exception as e:
        st.warning(f"Erreur PDF : {e}")
        st.info("Vérifiez que `fpdf2` est installé : `pip install fpdf2`")

    # ── Avertissement médical ─────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Avertissement médical :</strong>
    Ce système est une <strong>aide au diagnostic</strong> par intelligence artificielle.
    Il ne remplace pas l'examen clinique ni l'avis d'un ophtalmologiste qualifié.
    Toute décision thérapeutique doit être validée par un professionnel de santé habilité.
    </div>
    """, unsafe_allow_html=True)
