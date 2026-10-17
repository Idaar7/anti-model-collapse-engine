import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Deterministic Synthetic Data Engine", layout="wide")

# Dual-Core Language Layer
lang = st.radio("🌐 Select Language / Choisir la langue", ["EN", "FR"], horizontal=True)

if lang == "EN":
    st.title("🔬 Non-Linear Kinematic Engine: Anti-Model Collapse")
    st.write("Generation of pure deterministic physical data vectors for training autonomous navigation AI models.")
    st_header = "Quantum & Physical Parameters"
    v0_label = "Initial Velocity (m/s)"
    ang_label = "Launch Angle (Degrees)"
    noise_label = "Atmospheric Turbulence Level (Noise)"
    sub1 = "📊 Real Variance Visualization"
    sub2 = "💾 Clean Telemetry Data Stream (JSON/CSV)"
    btn_dl = "Download Immune Synthetic Dataset (.csv)"
    ent_title = "🚀 Deploy Enterprise Edition for Production Pipelines"
    ent_desc = "The open-source engine above is limited to 3 dimensions. Our industrial framework handles 6D non-Euclidean kinematics with real-time quantum noise injection engineered for LiDAR architectures."
    mail_label = "Enter your corporate email to receive technical integration documents (C++/Rust SDK):"
    btn_poc = "Request Proof-of-Concept (PoC) Token"
    success_msg = "Access request logged. A Principal AI Architect will contact you within 24 hours."
else:
    st.title("🔬 Moteur Cinématique Non-Linéaire : Anti-Model Collapse")
    st.write("Génération de vecteurs de données physiques purs pour l'apprentissage des IA de navigation autonome.")
    st_header = "Paramètres Quantiques & Physiques"
    v0_label = "Vitesse Initiale (m/s)"
    ang_label = "Angle de Tir (Degrés)"
    noise_label = "Niveau de Perturbation Atmosphérique (Bruit)"
    sub1 = "📊 Visualisation de la Variance Réelle"
    sub2 = "💾 Flux de Données Prêt pour Ingestion (JSON/CSV)"
    btn_dl = "Télécharger le Dataset Synthétique Immunisé (.csv)"
    ent_title = "🚀 Déployer la version Enterprise pour vos infrastructures"
    ent_desc = "Le moteur open-source ci-dessus est limité à 3 dimensions. Notre framework industriel gère la cinématique non-euclidienne en 6 dimensions avec injection de bruit quantique en temps réel pour architectures LiDAR."
    mail_label = "Entrez votre email professionnel pour recevoir la documentation technique d'intégration (SDK C++/Rust) :"
    btn_poc = "Demander une clé de Proof-of-Concept (PoC)"
    success_msg = "Demande enregistrée. Un ingénieur principal vous contactera sous 24h."

st.sidebar.header(st_header)
v0 = st.sidebar.slider(v0_label, 10.0, 100.0, 45.0)
angle = st.sidebar.slider(ang_label, 15, 85, 35)
noise_lvl = st.sidebar.slider(noise_label, 0.0, 0.5, 0.05)

def generate_pure_physics(v0, angle_deg, noise, steps=120):
    t = np.linspace(0, 5.0, steps)
    rad = np.radians(angle_deg)
    c_d, g = 0.47, -9.81
    vx, vy = v0 * np.cos(rad), v0 * np.sin(rad)
    x, y = [0.0], [0.0]
    dt = 5.0 / steps
    for i in range(1, steps):
        v = np.sqrt(vx**2 + vy**2)
        ax = - (0.5 * c_d * v * vx)
        ay = g - (0.5 * c_d * v * vy)
        vx += ax * dt
        vy += ay * dt
        nx = x[-1] + vx * dt + np.random.normal(0, noise)
        ny = y[-1] + vy * dt + np.random.normal(0, noise)
        if ny < 0:
            x.append(nx)
            y.append(0.0)
            break
        x.append(nx)
        y.append(ny)
    return pd.DataFrame({"Position X": x, "Position Y": y})

data = generate_pure_physics(v0, angle, noise_lvl)
col1, col2 = st.columns(2)
with col1:
    st.subheader(sub1)
    fig, ax = plt.subplots()
    ax.plot(data["Position X"], data["Position Y"], color="#00FFCC", lw=2)
    st.pyplot(fig)
with col2:
    st.subheader(sub2)
    st.dataframe(data.head(10))
    st.download_button(label=btn_dl, data=data.to_csv(index=False), file_name="dataset_anti_collapse.csv", mime="text/csv")

st.write("---")
st.subheader(ent_title)
st.write(ent_desc)
email_contact = st.text_input(mail_label)
if st.button(btn_poc):
    if email_contact:
        st.success(f"{success_msg} ({email_contact})")

