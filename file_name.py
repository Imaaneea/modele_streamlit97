import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

model = joblib.load(r'C:\Users\USER\Desktop\MSDE 6ème promotion\Module 6 Machine Learning\02_Labs\10_deploiement\Streamlit\app solution\modeliris6.pkl')

def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)  
    return prediction[0]

col1, col2 = st.columns([1, 2])

with col1:
    st.image(
        "https://th.bing.com/th/id/OIP.cLHuJrzjoysPx7Y8SH6FgAHaFY?w=550&h=400&rs=1&pid=ImgDetMain",
        use_column_width=True
    )
    
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.5)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)
    
    if st.button("Predict Flower Type"):
        prediction = predict_flower(sepal_length, sepal_width, petal_length, petal_width)
        st.write(prediction)

with col2:
    st.image(
        "https://th.bing.com/th/id/R.ebc3d0554e30879dee9ca5d29aabcd11?rik=5p27PimhhVvgwA&pid=ImgRaw&r=0",
        caption="Image Caption",
        use_column_width=False
    )
    st.title("MSDE6 : ML Course")
    st.header("Iris Flower Prediction APP")
    st.markdown("This App predicts the iris flower type")

    user_choice = st.selectbox(
        "How would you like to use the prediction model?",
        options=["Input parameters directly", "Load a file of data"]
    )
