import streamlit as st
import random

st.title("Data Science Prototype")
st.markdown("""
            ### Here text about DS progect
            ##### Some text

            
            
            """)

# UI
user_input = st.text_input("Enter some text", "")

# Functionality
def classify_text(text):
    categories = ['Positive', 'Negative', 'Neutral']
    return random.choice(categories)

if st.button("Classify"):
    result = classify_text(user_input)
    st.write("Classification Result:", result)