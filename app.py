import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os, glob, shutil

st.set_page_config(page_title="Universal Anomaly Detector - Industrie 4.0", page_icon="⚙️", layout="wide")

st.title("⚙️ Plateforme Universelle de Détection d'Anomalies Industrielles")
st.markdown("**Architecture Générique Sans Fuite de Données** | Comparaison Règle Métier Univariée vs Modèle Non Supervisé Multivarié (PCA SPE) | *Par Y. Hasnaoui & B. Filali*")
st.divider()

# ==============================================================================
# CHOIX DU MODE DANS LA BARRE LATÉRALE
# ==============================================================================
st.sidebar.header("🗂️ 1. Source des Données")
mode_source = st.sidebar.radio(
    "Choisir le jeu de données :",
    ["📊 Cas d'étude : Pompe Industrielle (Pré-configuré)", "📁 Mode Universel : Importer un fichier CSV"]
)

# ==============================================================================
# MODE 1 : CAS DE LA POMPE À EAU (PRÉ-CONFIGURÉ POUR LA SOUTENANCE)
# ==============================================================================
def trouver_ou_telecharger_csv():
    chemins_possibles = ['sensor.csv', '/content/sensor.csv', '/content/drive/MyDrive/sensor.csv']
    for c in chemins_possibles:
        if os.path.exists(c):
            return c
    recherche = glob.glob('/content/**/sensor.csv', recursive=True)
    if recherche:
        return recherche[0]
    import kagglehub
    path = kagglehub.dataset_download("nphantawee/pump-sensor-data")
    csv_source = os.path.join(path, "sensor.csv")
    CIBLE = "sensor.csv" if os.access(".", os.W_OK) else "/content/sensor.csv"
    shutil.copy(csv_source, CIBLE)
    return CIBLE

@st.cache_data
def charger_modele_pompe():
    chemin = trouver_ou_telecharger_csv()
    df = pd.read_csv(chemin)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    cols_drop = ['Unnamed: 0', 'sensor_15', 'sensor_50', 'sensor_51']
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])
    sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
    df[sensor_cols] = df[sensor_cols].ffill().bfill()

    capteurs_sensibles = ['sensor_00', 'sensor_01', 'sensor_02', 'sensor_03', 'sensor_04',
                          'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14']
    for col in capteurs_sensibles:
        df[f'{col}_std30'] = df[col].rolling(window=30, min_periods=1).std().fillna(0)
        df[f'{col}_diff15'] = df[col].diff(15).abs().fillna(0)

    features_opt = capteurs_sensibles + [f'{c}_std30' for c in capteurs_sensibles] + [f'{c}_diff15' for c in capteurs_sensibles]

    train_df = df.loc['2018-04-01 00:00:00':'2018-04-11 12:00:00'].copy()
    test_df  = df.loc['2018-04-11 12:01:00':'2018-04-20 00:00:00'].copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features_opt])
    X_test  = scaler.transform(test_df[features_opt])

    mu_ref, std_ref = train_df['sensor_04'].mean(), train_df['sensor_04'].std()
    test_df['score_metier'] = np.abs(test_df['sensor_04'] - mu_ref) / std_ref

    pca = PCA(n_components=5, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    erreur_par_var_test = (X_test - pca.inverse_transform(X_test_pca)) ** 2
    erreur_train = np.mean((X_train - pca.inverse_transform(X_train_pca)) ** 2, axis=1)
    erreur_test  = np.mean(erreur_par_var_test, axis=1)

    seuil_ref_sain = np.percentile(erreur_train, 99.5)
    test_df['score_modele'] = pd.Series(erreur_test / seuil_ref_sain, index=test_df.index).rolling(5, min_periods=1).mean()

    df_contrib = pd.DataFrame(erreur_par_var_test[:, :10], index=test_df.index, columns=capteurs_sensibles)
    return test_df, df_contrib

if mode_source.startswith("📊"):
    with st.spinner("Chargement du modèle Pompe Industrielle..."):
        test_df, df_contrib = charger_modele_pompe()

    def evaluer_pompe(df_eval, col_score, seuil, persistance_min, fenetre_pre_panne_heures=24):
        depassement = (df_eval[col_score] > seuil).astype(int)
        alerte = (depassement.rolling(window=persistance_min).sum() >= persistance_min).astype(int)
        est_en_reparation = (df_eval['machine_status'] != 'NORMAL').astype(int)
        zone_maint = (est_en_reparation.rolling(window=12*60, min_periods=1).max() == 1)
        alerte_exp = alerte.copy()
        alerte_exp[zone_maint & (df_eval['machine_status'] == 'RECOVERING')] = 0

        instants_pannes = df_eval[df_eval['machine_status'] == 'BROKEN'].index
        delais, zone_pre = [], pd.Series(False, index=df_eval.index)
        for t_panne in instants_pannes:
            deb = t_panne - pd.Timedelta(hours=fenetre_pre_panne_heures)
            zone_pre.loc[deb:t_panne] = True
            av = alerte.loc[deb:t_panne]
            delais.append((t_panne - av[av == 1].index[0]).total_seconds() / 60.0 if av.sum() > 0 else 0.0)

        zone_saine = (~zone_maint) & (~zone_pre)
        return alerte_exp, int((alerte[zone_saine].diff() == 1).sum()), int(alerte[zone_saine].sum()), delais

    st.sidebar.header("🎛️ 2. Réglages & Scénarios")
    if "seuil_mod" not in st.session_state:
        st.session_state.seuil_mod, st.session_state.pers, st.session_state.seuil_met = 0.90, 5, 3.5

    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("🔴 Tardive"):
        st.session_state.seuil_mod, st.session_state.pers, st.session_state.seuil_met = 1.20, 10, 3.5
    if c2.button("🟠 Fausse"):
        st.session_state.seuil_mod, st.session_state.pers, st.session_state.seuil_met = 0.50, 1, 3.5
    if c3.button("🟢 Optimal"):
        st.session_state.seuil_mod, st.session_state.pers, st.session_state.seuil_met = 0.90, 5, 3.5

    seuil_modele = st.sidebar.slider("Seuil Modèle PCA (SPE)", 0.40, 2.50, st.session_state.seuil_mod, 0.05)
    persistance_min = st.sidebar.slider("Persistance (minutes)", 1, 45, st.session_state.pers, 1)
    seuil_metier = st.sidebar.slider("Seuil Règle Métier (σ)", 1.5, 8.0, st.session_state.seuil_met, 0.5)

    alerte_m, fa_m, fp_m, delais_m = evaluer_pompe(test_df, 'score_metier', seuil_metier, persistance_min)
    alerte_if, fa_if, fp_if, delais_if = evaluer_pompe(test_df, 'score_modele', seuil_modele, persistance_min)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fausses Alertes (PCA)", f"{fa_if} épisode(s)", f"{fa_if - fa_m} vs Métier ({fa_m})", delta_color="inverse")
    k2.metric("Minutes Faux Positif (PCA)", f"{fp_if} min", f"{fp_if - fp_m} min vs Métier", delta_color="inverse")
    k3.metric("Anticipation Panne 1", f"{delais_if[0]:.0f} min ({delais_if[0]/60:.1f}h)", f"{delais_if[0] - delais_m[0]:+.0f} min vs Métier")
    k4.metric("Anticipation Panne 2", f"{delais_if[1]:.0f} min ({delais_if[1]/60:.1f}h)", f"{delais_if[1] - delais_m[1]:+.0f} min vs Métier")

    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    pannes = test_df[test_df['machine_status'] == 'BROKEN'].index
    rec = (test_df['machine_status'] == 'RECOVERING')

    axes[0].plot(test_df.index, test_df['score_metier'], color='steelblue', label='Écart Règle Métier (σ)')
    axes[0].axhline(seuil_metier, color='darkorange', linestyle='--', linewidth=2, label=f'Seuil ({seuil_metier}σ)')
    axes[0].fill_between(test_df.index, 0, 10, where=(alerte_m==1)&(~rec), color='orange', alpha=0.4, label='Alerte Métier')
    axes[0].fill_between(test_df.index, 0, 10, where=rec, color='gray', alpha=0.2, label='Maintenance (RECOVERING)')
    axes[0].set_ylim(0, 10)
    axes[0].set_title(f"1. RÈGLE MÉTIER UNIVARIÉE | Fausses alertes : {fa_m} | Anticipation Panne 1 : {delais_m[0]:.0f} min | Panne 2 : {delais_m[1]:.0f} min", fontweight='bold')

    axes[1].plot(test_df.index, test_df['score_modele'], color='purple', label='Score PCA Multivarié (SPE)')
    axes[1].axhline(seuil_modele, color='red', linestyle='--', linewidth=2, label=f'Seuil ({seuil_modele})')
    axes[1].fill_between(test_df.index, 0, 5, where=(alerte_if==1)&(~rec), color='red', alpha=0.4, label='Alerte PCA')
    axes[1].fill_between(test_df.index, 0, 5, where=rec, color='gray', alpha=0.2, label='Maintenance (RECOVERING)')
    axes[1].set_ylim(0, 5)
    axes[1].set_title(f"2. MODÈLE PCA MULTIVARIÉ | Fausses alertes : {fa_if} | Anticipation Panne 1 : {delais_if[0]:.0f} min | Panne 2 : {delais_if[1]:.0f} min", fontweight='bold')

    for ax in axes:
        for i, p in enumerate(pannes):
            ax.axvline(p, color='black', linewidth=2.5, linestyle=':', label='Casse (BROKEN)' if i==0 else "")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ==============================================================================
# MODE 2 : MODE UNIVERSEL (IMPORTER N'IMPORTE QUEL FICHIER CSV)
# ==============================================================================
else:
    st.info("💡 **Mode Universel activé :** Importez n'importe quel fichier CSV contenant des séries temporelles de capteurs. L'application construira automatiquement l'état de référence sans fuite de données, les indicateurs glissants et le modèle PCA.")
    fichier_upload = st.file_uploader("Glissez-déposez votre fichier de données (.csv)", type=["csv"])

    if fichier_upload is not None:
        df_custom = pd.read_csv(fichier_upload)
        st.write(f"✅ **Fichier chargé :** `{fichier_upload.name}` ({len(df_custom)} lignes, {len(df_custom.columns)} colonnes)")

        # Détection automatique des colonnes numériques et temporelles
        cols_toutes = list(df_custom.columns)
        cols_num = list(df_custom.select_dtypes(include=[np.number]).columns)

        st.sidebar.header("⚙️ 2. Configuration des Colonnes")
        col_temps = st.sidebar.selectbox("Colonne Temporelle (ou Index) :", ["Index automatique"] + cols_toutes)
        if col_temps != "Index automatique":
            try:
                df_custom[col_temps] = pd.to_datetime(df_custom[col_temps])
                df_custom = df_custom.set_index(col_temps).sort_index()
            except Exception:
                pass

        capteurs_choisis = st.sidebar.multiselect(
            "Capteurs à surveiller (PCA) :",
            options=cols_num,
            default=cols_num[:min(10, len(cols_num))]
        )
        capteur_metier = st.sidebar.selectbox("Capteur de la Règle Métier :", options=capteurs_choisis if capteurs_choisis else cols_num)

        st.sidebar.header("✂️ 3. Découpage & Paramètres")
        pct_train = st.sidebar.slider("Période de référence saine (% initial du dataset)", 10, 80, 40, 5)
        fenetre_glissante = st.sidebar.slider("Fenêtre glissante Feature Engineering (pas)", 5, 60, 20, 5)
        seuil_pca_u = st.sidebar.slider("Seuil d'anomalie PCA", 0.5, 5.0, 1.2, 0.1)
        seuil_met_u = st.sidebar.slider("Seuil Règle Métier (en σ)", 1.5, 6.0, 3.5, 0.5)
        pers_u = st.sidebar.slider("Persistance (pas de temps consécutifs)", 1, 30, 5, 1)

        if len(capteurs_choisis) >= 2:
            # Prétraitement causal anti-leakage
            df_work = df_custom[capteurs_choisis].copy()
            df_work = df_work.ffill().bfill().fillna(0)

            # Ajout automatique de la volatilité et dérivée
            for c in capteurs_choisis:
                df_work[f"{c}_std"] = df_work[c].rolling(window=fenetre_glissante, min_periods=1).std().fillna(0)
                df_work[f"{c}_diff"] = df_work[c].diff(max(1, fenetre_glissante//2)).abs().fillna(0)

            n_split = int(len(df_work) * (pct_train / 100.0))
            train_u = df_work.iloc[:n_split].copy()
            test_u  = df_work.iloc[n_split:].copy()

            scaler_u = StandardScaler()
            X_tr_u = scaler_u.fit_transform(train_u)
            X_te_u = scaler_u.transform(test_u)

            # Règle métier
            mu_u, std_u = train_u[capteur_metier].mean(), train_u[capteur_metier].std() + 1e-9
            test_u['score_metier'] = np.abs(test_u[capteur_metier] - mu_u) / std_u

            # PCA automatique
            n_comp = min(5, len(capteurs_choisis))
            pca_u = PCA(n_components=n_comp, random_state=42)
            X_tr_pca = pca_u.fit_transform(X_tr_u)
            X_te_pca = pca_u.transform(X_te_u)

            err_tr = np.mean((X_tr_u - pca_u.inverse_transform(X_tr_pca))**2, axis=1)
            err_te = np.mean((X_te_u - pca_u.inverse_transform(X_te_pca))**2, axis=1)
            ref_995 = np.percentile(err_tr, 99.5) + 1e-9
            test_u['score_pca'] = pd.Series(err_te / ref_995, index=test_u.index).rolling(3, min_periods=1).mean()

            # Alertes avec persistance
            al_m = ((test_u['score_metier'] > seuil_met_u).astype(int).rolling(pers_u).sum() >= pers_u).astype(int)
            al_p = ((test_u['score_pca'] > seuil_pca_u).astype(int).rolling(pers_u).sum() >= pers_u).astype(int)

            u1, u2, u3 = st.columns(3)
            u1.metric("Taille État Référence (Train)", f"{len(train_u)} points", f"Premiers {pct_train}% du fichier")
            u2.metric("Alertes Règle Métier (Test)", f"{int((al_m.diff()==1).sum())} épisode(s)", f"{int(al_m.sum())} points en alerte")
            u3.metric("Alertes Modèle PCA (Test)", f"{int((al_p.diff()==1).sum())} épisode(s)", f"{int(al_p.sum())} points en alerte")

            fig_u, axes_u = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
            axes_u[0].plot(test_u.index, test_u['score_metier'], color='steelblue', label=f'Écart {capteur_metier} (σ)')
            axes_u[0].axhline(seuil_met_u, color='darkorange', linestyle='--', label=f'Seuil ({seuil_met_u}σ)')
            axes_u[0].fill_between(test_u.index, 0, test_u['score_metier'].max(), where=al_m==1, color='orange', alpha=0.35, label='Alerte Métier')
            axes_u[0].set_title(f"1. RÈGLE MÉTIER UNIVARIÉE ({capteur_metier}) — Période de surveillance (Test)", fontweight='bold')
            axes_u[0].legend(loc='upper left')
            axes_u[0].grid(True, alpha=0.3)

            axes_u[1].plot(test_u.index, test_u['score_pca'], color='purple', label='Score Anomalie PCA (SPE)')
            axes_u[1].axhline(seuil_pca_u, color='red', linestyle='--', label=f'Seuil PCA ({seuil_pca_u})')
            axes_u[1].fill_between(test_u.index, 0, test_u['score_pca'].max(), where=al_p==1, color='red', alpha=0.35, label='Alerte PCA')
            axes_u[1].set_title(f"2. MODÈLE PCA MULTIVARIÉ ({len(capteurs_choisis)} capteurs + volatilité/dérivée) — Période de surveillance (Test)", fontweight='bold')
            axes_u[1].legend(loc='upper left')
            axes_u[1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig_u)

            # Bouton pour télécharger les résultats enrichis en CSV
            csv_export = test_u[['score_metier', 'score_pca']].copy()
            csv_export['alerte_metier'] = al_m
            csv_export['alerte_pca'] = al_p
            st.download_button(
                "📥 Télécharger le rapport d'alertes complet (.CSV)",
                data=csv_export.to_csv().encode('utf-8'),
                file_name="resultats_anomalies_export.csv",
                mime="text/csv"
            )
