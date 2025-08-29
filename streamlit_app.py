import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title('🤖 Penguines DataSet')

st.header('Machine learning app built with streamlit')

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')

exp =  st.expander('📃 Data Frame')
with exp:
    st.markdown('## Raw Data of Penguins DataSet')
    df
    st.markdown('---')
    st.markdown('## X Data')
    x_raw = df.drop("species", axis=1)
    x_raw
    st.markdown('---')
    st.markdown('## Y Data')
    y_raw = df["species"]
    y_raw
    st.markdown('---')

exp2 = st.expander('Data Visualisation')
with exp2:
    st.scatter_chart(data=df,x="bill_length_mm",y="body_mass_g",color="species")

with st.sidebar:
    st.markdown("# Input Penguin Data")
    st.text("Input Features")
    island = st.selectbox("Island", ("Biscoe", "Dream", "Torgersen"))
    bill_length_mm = st.slider("Bill Length (mm)", 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider("Bill Depth (mm)", 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider("Flipper Length (mm)", 172.0, 231.0, 201.0)
    body_mass_g = st.slider("Body Mass (g)", 2700.0, 6300.0, 4207.0)
    sex = st.selectbox("Gender", ("male", "female"))

#Data Frame for Sidebar Inputs
data = {'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': sex,
        }
input_df = pd.DataFrame(data,index=[0])
input_penguines = pd.concat([input_df, x_raw], axis=0)

with st.expander('Input Features Data Frames'):
    st.write("User Input Features")
    input_df
    st.write("Combined Input Features")
    input_penguines

#Data Perp
#Encode X
encode = ['island', 'sex']
df_penguines = pd.get_dummies(input_penguines, prefix=encode)
X = df_penguines[1:]
input_row = df_penguines[:1]

#Encode Y
target_mapper = {'Adelie':0,
                 'Chinstrap':1,
                 'Gentoo':2}

def target_encode(val):
    return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data Preperations'):
    st.write("**Encoded X (Input Penguine)**")
    input_row
    st.write("**Encoded Y**")
    y

#Model Training & Inference (Random Forest)
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to predict
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.rename(columns={0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}, inplace=True)
penguines_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])

#Display Results
st.markdown('## Prediction Results')

col1, col2, col3 = st.columns(3)
col1.metric("Adelie Probability", f"{prediction_proba[0][0]*100:.1f}%")
col2.metric("Chinstrap Probability", f"{prediction_proba[0][1]*100:.1f}%")
col3.metric("Gentoo Probability", f"{prediction_proba[0][2]*100:.1f}%")

st.dataframe(df_prediction_proba,
            column_config={
                'Adelie' : st.column_config.ProgressColumn(
                'Adelie',
                format='%f',
                width='medium',
                min_value=0,
                max_value=1
            ),'Chinstrap' : st.column_config.ProgressColumn(
                'Chinstrap',
                format='%f',
                width='medium',
                min_value=0,
                max_value=1
            ),'Gentoo' : st.column_config.ProgressColumn(
                'Gentoo',
                format='%f',
                width='medium',
                min_value=0,
                max_value=1
            )
            }, hide_index=True)
st.success(f"Species Name : {penguines_species[prediction][0]}")
