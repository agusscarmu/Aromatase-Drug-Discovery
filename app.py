import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import joblib

# Molecular descriptor calculator
def desc_calc():
    # Performs the descriptor calculation
    bashCommand = "java -Xms1G -Xmx1G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    # os.remove('molecule.smi')

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model building
def build_model(input_data, molecule):
    # Reads in saved regression model
    load_model = joblib.load('rfm.pkl')
    # Apply model to make predictions
    prediction = load_model.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(prediction, name='pIC50')
    molecule_name = pd.Series(molecule, name='molecule_name')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)

# # Logo image
# image = Image.open('logo.png')

# st.image(image, use_column_width=True)

# Page title
st.markdown("""
# Bioactivity Prediction App (Aromatase)

This app allows you to predict the bioactivity of Aromatase inhibitors. 
- **Python libraries:** 
    - Pandas
    - Streamlit
    - Scikit-learn
""")

def process_data(data):
    # Accede a la columna "0" del DataFrame y conviértela en una lista de cadenas
    lines = data[0].tolist()

    # Dividir las columnas por tabulaciones o espacios (ajustar según el delimitador real)
    df = pd.DataFrame([line.strip().split('\t') for line in lines], columns=['smile', 'molecule'])
    df.to_csv('molecule.smi', sep='\t', index=False, header=False)

    return df

# Sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
![Example input file](test.txt)
""")

if st.sidebar.button('Predict'):
    if uploaded_file is not None:
        load_data = pd.read_table(uploaded_file, sep=' ', header=None)
        data = process_data(load_data)
        st.success("File processed successfully.")
    else:
        st.warning("Please upload a file first.")

    st.header('**Original input data**')
    st.write(data)

    with st.spinner("Calculating descriptors..."):
        desc_calc()

    # Read in calculated descriptors and display the dataframe
    st.header('**Calculated molecular descriptors**')
    desc = pd.read_csv('descriptors_output.csv')
    st.write(desc)
    st.write(desc.shape)

    # Read descriptor list used in previously built model
    st.header('**Subset of descriptors from previously built models**')
    Xlist = list(pd.read_csv('descriptor_list.csv').columns)
    desc_subset = desc[Xlist]
    st.write(desc_subset)
    st.write(desc_subset.shape)

    # Apply trained model to make prediction on query compounds
    build_model(desc_subset, data['molecule'])
else:
    st.info('Upload input data in the sidebar to start!')