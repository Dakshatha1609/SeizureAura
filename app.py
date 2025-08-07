import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# LLM & Vectorstore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.llm import get_chatgroq_model
from models.embeddings import get_vectorstore_from_local

# Load env variables
load_dotenv()
st.set_page_config(page_title="SeizureAura AI Companion", layout="centered")
st.title(" SeizureAura - AI Health Companion")

st.sidebar.title(" Navigation")
page = st.sidebar.radio("Choose a page:", ["Seizure Risk Prediction", "Ask AI Chatbot"])

use_rag = use_web = False
if page == "Ask AI Chatbot":
    use_rag = st.sidebar.checkbox("Use Local Knowledge (RAG)", value=True)
    use_web = st.sidebar.checkbox("Enable Web Search Fallback", value=True)

# Seizure Risk Prediction Logic
if page == "Seizure Risk Prediction":
    uploaded_file = st.file_uploader(" Upload EEG File (CSV)", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(" File uploaded and previewed below")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f" File Error: {e}")

        if st.button(" Predict Seizure Risk"):
            with st.spinner("Predicting..."):
                try:
                    df = df.select_dtypes(include=[np.number]).dropna()
                    data = df.values
                    if data.shape[1] > data.shape[0]:
                        data = data.T
                    if data.shape[1] > 46:
                        data = data[:, :46]
                    if data.shape[1] != 46:
                        st.error(f"Model expects 46 features, found {data.shape[1]}")
                    else:
                        data = data.reshape(1, data.shape[0], data.shape[1])
                        from tensorflow.keras.models import load_model
                        model = load_model("seizure_model.keras", compile=False)
                        prediction = model.predict(data)[0][0]
                        result = "Seizure Risk" if prediction > 0.5 else "No Seizure Risk"
                        st.success(f"Prediction: **{result}**")
                        st.write(f"Probability: `{prediction:.2f}`")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

# Chatbot Logic
else:
    st.subheader(" Ask About Seizures or Symptoms")
    model = get_chatgroq_model()
    try:
        vectorstore = get_vectorstore_from_local()
    except Exception as e:
        st.warning(f"Could not load knowledge base: {e}")
        vectorstore = None

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your AI assistant. You can ask anything about seizures, aura stages, or symptoms."}
        ]

    mode = st.radio(" Response Mode", ["Concise", "Detailed"], horizontal=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a health question or symptom..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    context = ""
                    if vectorstore:
                        docs = vectorstore.similarity_search(prompt, k=2)
                        context = "\n\n".join([doc.page_content for doc in docs])

                    formatted_prompt = (
                        f"You are a medically aware chatbot for epilepsy patients.\n"
                        f"User Question: {prompt}\n"
                        f"Relevant Context: {context}\n"
                    )

                    if mode == "Concise":
                        formatted_prompt += (
                            "\n\nYou must answer in **2-3 lines max** using simple, layman terms. "
                            "Avoid medical jargon. Don't elaborate unless necessary."
                        )
                    else:
                        formatted_prompt += (
                            "\n\nGive a **detailed medical explanation**. Include possible causes, symptoms, risks, "
                            "and 2-3 lifestyle suggestions with examples. Use medical terminology where relevant."
                        )

                    reply = model.invoke([
                        SystemMessage(content="You are a friendly and medically-aware AI chatbot."),
                        HumanMessage(content=formatted_prompt)
                    ]).content
                except Exception as e:
                    reply = f"Sorry, I encountered an error: {e}"

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
