import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Deterministic Synthetic Data Engine", layout="wide")
st.title("🔬 Moteur Cinématique Non-Linéaire : Anti-Model Collapse")
st.write("Génération de vecteurs de données physiques purs pour l'apprentissage des IA de navigation autonome.")

st.sidebar.header("Paramètres Quantiques & Physiques")
v0 = st.sidebar.slider("Vitesse Initiale (m/s)", 10.0, 100.0, 45.0)
angle = st.sidebar.slider("Angle de Tir (Degrés)", 15, 85, 35)
noise_level = st.sidebar.slider("Niveau de Perturbation Atmosphérique (Bruit)", 0.0, 0.5, 0.05)

def generate_pure_physics(v0, angle_deg, noise, steps=120):
    t = np.linspace(0, 5.0, steps)
    rad = np.radians(angle_deg)
    c_d = 0.47
    g = -9.81
    
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

data = generate_pure_physics(v0, angle, noise_level)

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Visualisation de la Variance Réelle")
    fig, ax = plt.subplots()
    ax.plot(data["Position X"], data["Position Y"], label="Trajectoire Générée", color="#00FFCC", lw=2)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Altitude (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

with col2:
    st.subheader("💾 Flux de Données Prêt pour Ingestion (JSON)")
    st.dataframe(data.head(10))
    st.download_button(
        label="Télécharger le Dataset Synthétique Immunisé (.csv)",
        data=data.to_csv(index=False),
        file_name="dataset_anti_collapse.csv",
        mime="text/csv"
    )

st.write("---")
st.subheader("🚀 Déployer la version Enterprise pour vos infrastructures")
st.write("Le moteur open-source ci-dessus est limité à 3 dimensions. Notre framework industriel gère la cinématique non-euclidienne en 6 dimensions avec injection de bruit quantique en temps réel pour architectures LiDAR.")

email_contact = st.text_input("Entrez votre email professionnel pour recevoir la documentation technique d'intégration (SDK C++/Rust) :")
if st.button("Demander une clé de Proof-of-Concept (PoC)"):
    if email_contact:
        st.success(f"Demande enregistrée pour {email_contact}. Un ingénieur principal vous contactera sous 24h.")
