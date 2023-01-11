import numpy as np
import pickle
import streamlit as st
import base64

# load saved models
def load():
    global __model
    global __cv

    with open("./stress.pickle", "rb") as f:
        __model = pickle.load(f)
    
    with open('./count_vectorizer.pkl', 'rb') as f:
        __cv = pickle.load(f)

def det_stress(user):
    data = __cv.transform([user]).toarray()
    prob = np.round(__model.predict_proba(data) * 100, 2).tolist()[0]

    return f"not stressed: {prob[0]}%, stressed: {prob[1]}%" 

@st.experimental_memo

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("s.jpeg")

page_bg_img = f""" 
<style>
[data-testid="stAppViewContainer"]> .main {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid= "stHeader"]{{
background: rgba(0,0,0,0);
}}

</style>
"""

def main():
    #  background image
    st.markdown(page_bg_img, unsafe_allow_html=True)


    # Title
    st.title("Stress Detector")

    # user instructions
    instructions = '<p style="font-family:Courier; color:White; font-size: 20px;">get to know you mental health status at the touch of a button</p>'
    st.markdown(instructions, unsafe_allow_html=True)

    #user input
    user = st.text_input("Enter what you feel")

    detect = ''

    if st.button("DETECT"):
        detect = det_stress(user)
    
    st.success(detect)

    
    conclusion = '<p style="font-family:Courier; color:White; font-size: 20px;"></p>'
    st.markdown(conclusion, unsafe_allow_html=True)


if __name__ == "__main__":
    load()
    main()
