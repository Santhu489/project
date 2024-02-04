import streamlit as st
import pandas as pd
from prediction import predict

st.title('Autism  Genes Classifier')
st.markdown("A simple web app to classify genes as syndromic or non-syndromic.")

# Load the data
genes = pd.read_csv("sfari_genes.csv")

# Get user input for a gene symbol
gene_symbol = st.text_input("Enter a gene symbol:")

if st.button("Classify Gene"):
    if gene_symbol in genes['gene-symbol'].values:
        # Extract the corresponding row from the dataframe
        gene_info = genes[genes['gene-symbol'] == gene_symbol]

        # Classify the gene as syndromic or non-syndromic
        syndromic_status = predict(gene_info)

        if syndromic_status == 1:
            st.subheader("Classification Result")
            st.write(f"The gene {gene_symbol} is associated with autism.")
        else:
            st.subheader("Classification Result")
            st.write(f"The gene {gene_symbol} is not associated with autism.")
    else:
        st.write("The gene symbol does not exist in the data.")
