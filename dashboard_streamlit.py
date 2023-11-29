import streamlit as st
from PIL import Image

# Page icon
##icon = Image.open('img/loan_application.png')

@st.cache_data
def main():
    st.header("Défault de paiement - Machine Learning Project")
    st.markdown("Déploiement du modèle d'apprentissage automatique de l'ensemble de données sur le risque de défaut de crédit immobilier à l'aide de la regression logistque.")
    st.markdown("Utilisez ce tableau de bord pour comprendre les données et utiliser le modèle de prediction.")
    st.markdown("")
    ###st.image("/project_7/loan_application.jpeg")

if __name__ == '__main__':
    st.set_page_config(page_title="Modèle de scoring - Défault de paiement")
    main()