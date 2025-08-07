import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from llm import get_chatgroq_model
from embeddings import get_vectorstore_from_local
from utils import search_web

load_dotenv()
st.set_page_config(page_title="SeizureAura AI Companion", layout="centered")
st.title(" SeizureAura - AI Health Companion")

st.sidebar.title(" Navigation")
page = st.sidebar.radio("Choose a page:", ["Seizure Risk Prediction", "Ask AI Chatbot"])
use_rag = st.sidebar.checkbox("Use Local Knowledge (RAG)", value=True) if page == "Ask AI Chatbot" else False
use_web = st.sidebar.checkbox("Enable Web Search Fallback", value=True) if page == "Ask AI Chatbot" else False

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("seizure_model.keras")

def preprocess_eeg(df):
    df = df.select_dtypes(include=[np.number]).dropna()
    data = df.values
    if data.shape[1] > data.shape[0]:
        data = data.T
    if data.shape[1] > 46:
        data = data[:, :46]
    if data.shape[1] != 46:
        raise ValueError(f"Model expects 46 features, found {data.shape[1]}")
    data = data.reshape(1, data.shape[0], data.shape[1])
    return data

if page == "Seizure Risk Prediction":
    uploaded_file = st.file_uploader("Upload EEG File (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.button("Predict Seizure Risk"):
            try:
                model = load_model()
                data = preprocess_eeg(df)
                prediction = model.predict(data)[0][0]
                result = "Seizure Risk" if prediction > 0.5 else "No Seizure Risk"
                st.success(f"Prediction: **{result}**")
                st.write(f"Probability: `{prediction:.2f}`")

                model_llm = get_chatgroq_model()
                prompt = f"What does '{result}' mean for an epilepsy patient? Suggest 2 actions."
                if use_rag:
                    vectorstore = get_vectorstore_from_local()
                    context = "\n".join([doc.page_content for doc in vectorstore.similarity_search(prompt, k=2)])
                    prompt += f"\n\nContext:\n{context}"

                response = model_llm.invoke([
                    SystemMessage(content="You are a medical assistant."),
                    HumanMessage(content=prompt)
                ])
                st.subheader("AI Explanation")
                st.info(response.content)
            except Exception as e:
                st.error(f"Prediction Error: {e}")

else:
    st.subheader("Ask About Seizures or Symptoms")
    model = get_chatgroq_model()

    try:
        vectorstore = get_vectorstore_from_local()
    except:
        vectorstore = None
        st.warning("Knowledge base not loaded.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Ask anything about seizures, aura, or epilepsy symptoms."}
        ]

    mode = st.radio("Response Mode", ["Concise", "Detailed"], horizontal=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a health question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                context = ""
                if use_rag and vectorstore:
                    docs = vectorstore.similarity_search(prompt, k=2)
                    context = "\n".join([doc.page_content for doc in docs]) if docs else ""
                elif use_web:
                    context = search_web(prompt)

                final_prompt = (
                    f"You are a helpful health assistant.\n\n"
                    f"User question: {prompt}\n\n"
                    f"Relevant context:\n{context}\n\n"
                    f"Please respond {'in 2-3 lines' if mode == 'Concise' else 'with examples and medical insights.'}"
                )

                reply = model.invoke([
                    SystemMessage(content="You are a medically aware AI chatbot."),
                    HumanMessage(content=final_prompt)
                ]).content
            except Exception as e:
                reply = f"Error: {e}"

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
