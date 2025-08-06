import requests
import streamlit as st
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from model.predict import run_model 

load_dotenv()
print(" GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './models')))
from llm import get_chatgroq_model
from embeddings import get_vectorstore_from_local

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './utils')))
from utils import answer_from_local_knowledge, search_web

st.set_page_config(page_title="SeizureAura AI Companion", layout="centered")
st.title(" SeizureAura - AI Health Companion")

st.sidebar.title(" Navigation")
page = st.sidebar.radio("Choose a page:", ["Seizure Risk Prediction", "Ask AI Chatbot"])
if page == "Ask AI Chatbot":
    use_rag = st.sidebar.checkbox("Use Local Knowledge (RAG)", value=True)
    use_web = st.sidebar.checkbox("Enable Web Search Fallback", value=True)
else:
    use_rag = False
    use_web = False

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
            with st.spinner("Predicting from model..."):
                try:
                    res = run_model(uploaded_file)
                    if "result" not in res:
                        st.error("Model did not return 'result'.")
                    else:
                        st.success(f" Prediction: **{res['result']}**")
                        st.write(f" Probability: `{res['probability']:.2f}`")

                        if use_rag:
                            st.subheader(" Explanation from Local Knowledge")
                            prompt = f"What does '{res['result']}' mean for an epilepsy patient? Give 2 suggestions."
                            explanation = answer_from_local_knowledge(prompt)
                        else:
                            model = get_chatgroq_model()
                            prompt = f"Explain what '{res['result']}' means for an epilepsy patient in simple terms. Add 2 lifestyle suggestions."
                            explanation = model.invoke([
                                SystemMessage(content="You are a helpful medical assistant."),
                                HumanMessage(content=prompt)
                            ]).content

                        st.subheader(" AI Explanation")
                        st.info(explanation)
                except Exception as e:
                    st.error(f" Prediction Error: {e}")
else:
    st.subheader(" Ask About Seizures or Symptoms")
    model = get_chatgroq_model()
    try:
        vectorstore = get_vectorstore_from_local()
    except Exception as e:
        st.warning(f" Could not load knowledge base: {e}")
        vectorstore = None

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
                    if not vectorstore:
                        raise ValueError("Vectorstore not loaded.")

                    docs = vectorstore.similarity_search(prompt, k=2)
                    if not docs:
                        raise ValueError("No relevant documents found.")
                    context = "\n\n".join([doc.page_content for doc in docs])
                except Exception as e:
                    context = ""
                    st.warning(f"Error during RAG: {e}")

                try:
                    if not context and use_web:
                        web_context = search_web(prompt)
                        context = web_context if web_context else "No reliable web data found."

                    formatted_prompt = (
                        f"You are a medically aware health chatbot for epilepsy patients. "
                        f"Below is a user question followed by context from medical knowledge.\n\n"
                        f"User Question: {prompt}\n\n"
                        f"Relevant Context:\n{context}\n\n"
                    )

                    if mode == "Concise":
                        formatted_prompt += "Please respond concisely in 2-3 lines."
                    else:
                        formatted_prompt += "Please explain in detail with examples and medical facts."

                    reply = model.invoke([
                        SystemMessage(content="You are a friendly and medically-aware AI chatbot."),
                        HumanMessage(content=formatted_prompt)
                    ]).content
                except Exception as e:
                    reply = f"Sorry, I encountered an error: {e}"

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
