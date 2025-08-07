import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from dotenv import load_dotenv

# Add folders to path
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "config"))

# Load environment variables
load_dotenv()

# Imports
from models.llm import get_chatgroq_model
from models.embeddings import get_vectorstore_from_local
from models.seizure_model import SeizurePredictionModel
from utils.web_search import search_web
from langchain_core.messages import HumanMessage, SystemMessage
from tensorflow.keras.models import load_model
from keras.utils import custom_object_scope
import tensorflow as tf


# Streamlit setup
st.set_page_config(page_title="SeizureAura AI Companion", layout="centered")
st.title(" SeizureAura - AI Health Companion")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Seizure Risk Prediction", "Ask AI Chatbot"])

use_rag = use_web = False
if page == "Ask AI Chatbot":
    use_rag = st.sidebar.checkbox("Use Local Knowledge (RAG)", value=True)
    use_web = st.sidebar.checkbox("Enable Web Search Fallback", value=True)

# ---------------------------------------------------
# SEIZURE RISK PREDICTION
# ---------------------------------------------------
if page == "Seizure Risk Prediction":
    uploaded_file = st.file_uploader("Upload EEG CSV File", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

        if st.button("Predict Seizure Risk"):
            try:
                df = df.select_dtypes(include=[np.number]).dropna()
                data = df.values

                if data.shape[1] > data.shape[0]:
                    data = data.T
                if data.shape[1] > 46:
                    data = data[:, :46]
                if data.shape[1] != 46:
                    st.error(f"Expected 46 features, found {data.shape[1]}")
                else:
                    data = data.reshape(1, data.shape[0], data.shape[1])

                    with custom_object_scope({'SeizurePredictionModel': SeizurePredictionModel}):
                        model = load_model(
                            "seizure_model_cleaned.keras",
                            compile=False,
                            custom_objects={"SeizurePredictionModel": SeizurePredictionModel}
                        )
                    prediction = model.predict(data)[0][0]
                    result = " Seizure Risk" if prediction > 0.5 else " No Seizure Risk"
                    st.success(f"**Prediction:** {result}")
                    st.write(f"**Probability:** `{prediction:.2f}`")

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ---------------------------------------------------
# AI CHATBOT SECTION
# ---------------------------------------------------
else:
    st.subheader(" Ask About Seizures, Aura, or Symptoms")

    try:
        model = get_chatgroq_model()
    except Exception as e:
        st.error(f"LLM init error: {e}")
        st.stop()

    try:
        vectorstore = get_vectorstore_from_local()
    except Exception as e:
        st.warning(f"Knowledge base load error: {e}")
        vectorstore = None

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your AI assistant. Ask anything about seizures or aura symptoms."}
        ]

    mode = st.radio("Response Mode", ["Concise", "Detailed"], horizontal=True)

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
                    context_source = ""
                    if use_rag and vectorstore:
                        docs = vectorstore.similarity_search(prompt, k=2)
                        context = "\n\n".join([doc.page_content for doc in docs])
                        context_source = "Based on the local medical knowledge base, here's what I found."
                    elif use_web:
                        context = search_web(prompt) or ""
                        context_source = "Based on recent information found online, here's what I found."
                    else:
                        context = ""
                        context_source = "Answering from general medical knowledge."

                    if mode == "Concise":
                        system_message = "You are a concise, medically-aware assistant. Respond in 2-3 lines using simple, easy-to-understand language."
                        full_prompt = (
                            f"{context_source}\n\n"
                            f"User Question: {prompt}\n\n"
                            f"Context:\n{context}\n\n"
                            f"Please give a short, clear answer using layman-friendly terms."
                        )
                    else:
                        system_message = "You are a detailed, medically-aware assistant. Respond with causes, risks, and practical tips while staying understandable."
                        full_prompt = (
                            f"{context_source}\n\n"
                            f"User Question: {prompt}\n\n"
                            f"Context:\n{context}\n\n"
                            f"Give a complete explanation with 2-3 practical lifestyle tips or suggestions."
                        )

                    reply = model.invoke([
                        SystemMessage(content=system_message),
                        HumanMessage(content=full_prompt)
                    ]).content

                except Exception as e:
                    reply = f"Sorry, I encountered an error: {e}"

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})