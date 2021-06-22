import streamlit as st
import pandas as pd
import squarify
import matplotlib.pyplot as plt
import joblib,os
from PIL import Image
from pathlib import Path
# Vectorizer
news_vectorizer = open("resources/vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

def main():
    st.title("Sentiment Classifier")
    options = ["Home","Classify Tweet","Manage Tweets","Help"]
    choice = st.sidebar.selectbox("",options)

    # layout
    l1,l2 = st.beta_columns(2) #create two column layouts

    def read_markdown_file(markdown_file):
        return Path(markdown_file).read_text()

    #read data
    df = pd.read_csv("resources/train.csv")
    if choice == "Help":
        #st.subheader("Help")
        with st.beta_expander("Instructions"):
            video_file = open("resources/help_video.webm",'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)


    elif choice == "Classify Tweet":
        st.subheader("Classify Tweet")
        with st.beta_expander("Tweet"):
            tweet_text = st.text_area("Type Here")
            st.write(tweet_text)

            if st.button("Classify"):
                # Transforming user input with vectorizer
                vect_text = tweet_cv.transform([tweet_text]).toarray()
                predictor = joblib.load(open(os.path.join("resources/model_1.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                st.success("Tweet classified as:{}".format(prediction))

    elif choice == "Manage Tweets":
        #st.subheader("Manage Tweets")
        with l2:
            with st.beta_expander("Raw Tweets"):
                st.write(df[['sentiment', 'message']])


        with l1:
            with st.beta_expander("Distribution Of Tweets Per Sentiment",expanded = True):
                fig,ax=plt.subplots()
                ax.hist(df["sentiment"])
                st.pyplot(fig)




    else:
        # read image and mark down file
        markdown_file_path = read_markdown_file("resources/about.md")
           
        with st.beta_expander("About",expanded = True):
            img = Image.open("resources/imgs/climate_change.jpeg")
            st.image(img,use_column_width=True)
            st.markdown(markdown_file_path,unsafe_allow_html=True)


        

if __name__=='__main__':
    main()