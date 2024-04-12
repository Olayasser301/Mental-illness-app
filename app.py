import streamlit as st
from streamlit_option_menu import option_menu
import joblib

MODEL_PERCEPTRON = "preceptron"

def main():
    show_header = True
    if "current_page" in st.session_state and st.session_state["current_page"] == "submit_page":
        show_header = False

    if show_header:
        st.title("Mental illness prediction App")
        page = option_menu(
            "Select a page", ["Take Quiz", "About", "Further Reading"], 
            orientation="horizontal", icons=["puzzle", "house-door", "book-half"],
            key="current_page")
    else:
        page = st.session_state["current_page"]

    if st.session_state["current_page"]:
        page = st.session_state["current_page"]

    if page == "Take Quiz":
        show_Quiz()

    elif page == "About":
        show_about_page()

    elif page == "Further Reading":
        show_Further_Reading_page()

    elif page == "submit_page":
        show_submit_page()


def show_Quiz():
    st.title("Mental Illness Prediction Quiz")
    model_options = ["Support vector machine", "xgboost RF", "preceptron", "Neural Networks"]
    model_selection = st.selectbox("Select a Model", model_options)

    if model_selection == "Support vector machine":
        show_Quiz_for_model("Support vector machine")

    elif model_selection == "xgboost RF":
        show_Quiz_for_model("xgboost RF")

    elif model_selection == "preceptron":
        show_Quiz_for_model("preceptron")

    elif model_selection == "Neural Networks":
        show_Quiz_for_model("Neural Networks")


def show_Quiz_for_model(model_name):
    st.title(f"Mental Illness Prediction Quiz for {model_name}")

    st.markdown("Check all the boxes that applies")
    sex = st.selectbox(label="Sex", options=["Male", "Female", "Diverse"] , key ="sex")
    is_self_employed = st.checkbox(label="Are you self employed?", key="is_self_employed")
    st.markdown("if no then")
    is_comp_employed = st.selectbox(label = "what is your company size? ", options=["1-5" , "6-25", "26-100" , "100-500" ,"500-1000", ">1000"] , key="comp_size")
    tech_comp_flag = st.checkbox(label = " Do you work in a tech company?" , key = "work_in_tech")
    mh_family_hist = st.selectbox(label= "Does your family have a history of a mental illness?", options=["No", "Yes", "I don't know"] , key = "mh-fam")
    mh_disorder_past = st.selectbox(label= " Do YOU have a history with mental illnesses?", options=["No", "Yes", "Maybe"], key = "mh_hist")
    mh_diagnos_proffesional = st.selectbox(label= "were you diagnosed professionally before?", options=["No", "Yes"], key = "diag_prof")
    st.markdown(f"""### Form Details
   ```
   Sex: {sex}
   Are you self-employed? {is_self_employed}
   if no then
   What is your company size? {is_comp_employed}
   Do you work in a tech company?{tech_comp_flag}
   family history? : {mh_family_hist}
   mental health history? : {mh_disorder_past}
   diagnosed professionally? : {mh_diagnos_proffesional}
   ```
   """)

    st.button("Submit", on_click=lambda: submit_on_click(model_name))


def submit_on_click(model_name):
    st.session_state["current_model"] = model_name
    st.session_state["current_page"] = f"submit_page"


def show_submit_page():
    model_name = st.session_state["current_model"]
    features, labels, model = load_model(model_name)

    X = {
       'self_empl_flag': 1 if st.session_state["is_self_employed"] else 0,
       'comp_no_empl': st.session_state["comp_size"] ,
       'sex': st.session_state["sex"],
       'tech_comp_flag': 1 if st.session_state["work_in_tech"] else 0,
       'mh_family_hist': st.session_state["mh-fam"],
       'mh_disorder_past': st.session_state["mh_hist"],
       'mh_diagnos_proffesional': st.session_state["diag_prof"]
   }
    X = [list(X.values())]
    X = features.transform(X)

    y = model.predict(X)
    result = labels.inverse_transform(y)[0]

    if result == "Yes":
       st.title("Congrats, you are mentally ill <3!")
       VIDEO_URL = "https://www.youtube.com/watch?v=mBw3qzf4s18"
       st.video(VIDEO_URL, subtitles= None)
       st.balloons()
       #st.audio("/home/work/Downloads/50_cent_in_da_club.mp3", format="audio/mpeg", loop=True)
  
    elif result == "No":
       st.title("Damn better luck next time :(")
       VIDEO_URL = "https://www.youtube.com/watch?v=8Uerp99QSSU"
       st.video(VIDEO_URL, subtitles= None)
    elif result == "Maybe":
       st.title("Welp, we think you might be sick , praying for you<3!")

    if model_name != MODEL_PERCEPTRON:
        prob = model.predict_proba(X)
        st.markdown(f"Here are your chances:")
        for label, probability in zip(labels.classes_, prob[0]):
            st.markdown(f"- {label}: {round(probability * 100, 2)}%")
  
    st.button("go back!", on_click=goback_on_click)

   
def load_model(model_name):
    if model_name == "Support vector machine":
        features = joblib.load("/home/work/Downloads/SVCF.joblib")
        labels = joblib.load("/home/work/Downloads/SVCL.joblib")
        model = joblib.load("/home/work/Downloads/SVCM.joblib")
        pass
    elif model_name == "xgboost RF":
       features = joblib.load("/home/work/Downloads/features_encoder-v2.joblib")
       labels = joblib.load("/home/work/Downloads/labels_encoder-v2.joblib")
       model = joblib.load("/home/work/Downloads/model-v3.joblib")
       pass
    elif model_name == MODEL_PERCEPTRON:
        features = joblib.load("/home/work/Downloads/precepF.joblib")
        labels = joblib.load("/home/work/Downloads/precepL.joblib")
        model = joblib.load("/home/work/Downloads/precepM2.joblib")    
        pass
    elif model_name == "Neural Networks": #This is the best model so far than the RF where NN is 75% , RF is 73.4% and SVC 71.8% while perceptron is 69.2%
        features = joblib.load("/home/work/Downloads/NNF3.joblib")
        labels = joblib.load("/home/work/Downloads/NNL3.joblib")
        model = joblib.load("/home/work/Downloads/NNM3.joblib")
        pass
    else:
        raise Exception(f"Given name does not exist: {model_name}")

    return features, labels, model


def goback_on_click():
    st.session_state["current_page"] = "Take Quiz"

def show_about_page():
   st.header("About Page")
   st.header("About Mental Illness Prediction")
   st.write("Welcome to our website dedicated to mental illness prediction. We are committed to providing valuable insights and resources to help individuals better understand and manage their mental health.")
   st.header("Our Mission")
   st.write("Our mission is to leverage the power of data science and machine learning to develop accurate predictive models for various mental illnesses. By analyzing diverse datasets and employing advanced algorithms, we aim to create tools that can assist healthcare professionals in early detection, diagnosis, and treatment planning.")
   st.header("What We Offer")
   st.write("Predictive Models: We develop predictive models that can identify patterns and risk factors associated with different mental health conditions.")
   st.write("Educational Resources: We provide educational resources to increase awareness and understanding of mental health issues, including articles, videos, and infographics.")
   st.write("Supportive Community: We foster a supportive online community where individuals can share their experiences, seek advice, and find solidarity in their mental health journey.")
   st.header("Our Team")
   st.write("Our team consists of passionate individuals with expertise in data science, psychology, psychiatry, and mental health advocacy. We are dedicated to making a positive impact on mental health outcomes through innovative research and technology.")
   st.header("Meet our Team")
   st.image("sora1.jpeg")
   st.header("Get Involved")
   st.write("Join us in our mission to improve mental health prediction and support. Here are some ways you can get involved:")
   st.write("Contribute Data: If you have access to anonymized mental health datasets, consider contributing them to our research efforts.")
   st.write("Share Your Story: Share your personal experiences with mental illness to help reduce stigma and raise awareness.")
   st.write("Volunteer: Volunteer your time and skills to help with data analysis, software development, or community moderation.")
   st.header("Contact Us")
   st.write("Have questions or feedback? We'd love to hear from you! Feel free to reach out to us via email at contact@mentalillnessprediction.com")
def show_Further_Reading_page():
   st.header("Educational Resources")
   st.header("Articles")
   st.markdown("Understanding Mental Illness: [https://www.uchicagomedicine.org/forefront/health-and-wellness-articles/2022/may/mental-health-awareness-q-and-a]")
   st.markdown("The Role of Predictive Modeling in Mental Health: [https://jest.com.pk/index.php/jest/article/view/72]")
   st.header("Videos")
   st.markdown("Mental Health Awareness Videos: [https://www.youtube.com/watch?v=eWZ_8iQi59c]")
   st.markdown("Explainer Videos on Predictive Models: [https://www.youtube.com/watch?v=JOArz7wggkQ]")
   st.header("Infographics")
   st.markdown("Infographics on Mental Health Statistics: [https://www.nami.org/About-Mental-Illness/Mental-Health-by-the-Numbers/Infographics-Fact-Sheets]")
   st.markdown("Infographics on Risk Factors: [https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response/?gad_source=1&gclid=CjwKCAjwt-OwBhBnEiwAgwzrUl60cOhvT6kfb8B3j7V0H5DtLp7g_apn_Qt8oToz74wYO5i65n7pMRoCNhQQAvD_BwE]")
   st.header("Recommended Reading List")
   st.markdown("Books on Mental Health: [https://www.amazon.com/Anxious-Generation-Rewiring-Childhood-Epidemic/dp/0593655036/ref=zg_bs_g_4682_d_sccl_1/141-1010307-2091314?psc=1]")
   st.markdown("Books on Predictive Modeling: [https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715]")

if __name__ == "__main__":
    main()

