import pickle
from sklearn.feature_extraction.text import CountVectorizer
from gtts import gTTS
import streamlit as st

model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vectorized.pkl', 'rb'))



def main():
    st.title("Email Spam Classification Application")
    st.write("Build with Streamlit & Python")
    activ = ["Classification", "About"]
    choices = st.sidebar.selectbox("Select Activities", activ)
    if choices == "Classification":
        st.subheader("Classification")
        msg = st.text_input("Enter a text")
        if st.button("Process"):
            print(msg)
            print(type(msg))
            data = [msg]
            print(data)
            vec = cv.transform(data).toarray()
            result = model.predict(vec)
            if result[0] == 0:
                st.success("This is Not A Spam Email")

            else:
                st.error("This is A Spam Email")


main()
