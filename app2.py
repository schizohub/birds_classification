import streamlit as st 
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pandas as pd
import numpy as np

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

plt = platform.system()
if plt=='Linux': pathlib.WindowsPath= pathlib.PosixPath


st.title('Classification of some bird species')
st.write('Would you like to upload birds images in order to know how my deep learning model works and its prediction of accuracy?. If yes, then let\'s go!')

st.write('Now, you can be able to upload some species pictures that I used to build this model, so there are:')
a = ['Canary', 'Chicken', 'Duck', 'Eagle', 'Falcon', 'Goose','Magpie', 'Owl', 'Penguin', 'Raven', 'Sparrow', 'Swan', 'Turkey', 'Woodpecker']
df = pd.DataFrame(a, index=range(1,15),columns=['Birds'])
st.dataframe(df, height=500,width=275)
st.write("So you must send those species pictures!")
file = st.file_uploader('Upload picture', type=['jpeg','png','gif','svg','jpg'])
if file:
        st.image(file)

        img = PILImage.create(file)

        model = load_learner('bird_class_resnet50.pkl')

        pred,pred_id,probs = model.predict(img)
        st.success(f"Prediction: {pred}")
        st.info(f"Probability: {probs[pred_id]*100:.1f}%")

        # fig = px.bar(y= model.dls.vocab,x=probs*100)
        # st.plotly_chart(fig)
        arr = np.array(probs*100)
       
        b = []
        c = []
        for i in arr:
            b.append(f"{np.round(i)}%")
        for i in arr:
            c.append(i)
        arr = np.array(b)
        arr1 = np.array(c)
        df = pd.DataFrame([arr],columns=['Canary', 'Chicken', 'Duck', 'Eagle', 'Falcon', 'Goose','Magpie', 'Owl', 'Penguin', 'Raven', 'Sparrow', 'Swan', 'Turkey', 'Woodpecker'])
        st.text('This is prediction table of image that you sent')
        st.table(df.T)
        df1 = pd.DataFrame([arr1], columns=df.columns)
        # figma = sns.barplot(x=df.T.index, y=df1.T.values)
        st.bar_chart(df1.T)        
        source = pd.DataFrame({
            'Birds': ['Canary', 'Chicken', 'Duck', 'Eagle', 'Falcon', 'Goose','Magpie', 'Owl', 'Penguin', 'Raven', 'Sparrow', 'Swan', 'Turkey', 'Woodpecker'],
            'Prediction': arr1
        })
        figma = alt.Chart(source).mark_bar().encode(
            x='Birds',
            y='Prediction'
        )
        st.altair_chart(figma)
        # st.write(figma)

        st.write("Actually this is my second deep learning project kinda real project, so that it could be some failure to differentiate images. Firstly I celled once my code on colab, it showed 70% ,accuracy in RESNET34 architecture, and VGG16 architechture has shown 71 %, and I wrote RESNET50 architechture, and BOOM!!!💥💥💥 Yeah, it's eventually released low accuracy but a little more rather than other arch. That's why I chose RESNET50 arch.,cause of being accuracy has shown 73%")
        st.markdown('Soon, **_some news upcoming!!!_ I\'m gonna return new projects, till then**, see around.')




