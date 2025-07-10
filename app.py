import streamlit as st 
import joblib

vectorizer = joblib.load("vectorizer.jb")
model=joblib.load("lr_model.jb")

st.title("fake news Detector")
st.write("Enter a News Article below to check whether it is fake or real. ")

news_input =st.text_area("News Article:","")

if st.button("Check News"):
    if news_input.strip():
        transform_input =vectorizer.transform([news_input])
        prediction =model.predict(transform_input)
        
        if prediction[0]==1:
            st.success("The News is Real! ")
        else:
            st.error("The News is Fake!")
    else:
        st.warning("please enter some text to analyze. ")
            