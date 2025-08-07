SeizureAura - AI Health Companion 



This AI-powered chatbot helps predict seizure risk using EEG data and provides intelligent responses related to epilepsy, aura stages, symptoms, and lifestyle advice.



Features:

\-  Upload EEG files (CSV) and get real-time seizure risk predictions

\-  Ask health-related questions about epilepsy, aura stages, symptoms

\-  RAG: Local Knowledge + Web Search + LLM fallback

\-  Toggle between Concise and Detailed answers

\-  Powered by GROQ LLM (`llama3-8b-8192`)



Tech Stack



\- Frontend: Streamlit

\- Model: TensorFlow CNN + BiLSTM

\- RAG: LangChain + HuggingFace + FAISS

\- LLM: Groq `llama3-8b-8192`

\- Deployment: GitHub + Streamlit Cloud




Setup Instructions



To run this project locally:



Step 1: Clone the repo

git clone https://github.com/Dakshatha1609/SeizureAura.git

cd SeizureAura



Step 2: Install dependencies

pip install -r requirements.txt



Step 3: Run the app

streamlit run app.py



Links



\- GitHub Repo: \[https://github.com/Dakshatha1609/SeizureAura](https://github.com/Dakshatha1609/SeizureAura)

\- Live Streamlit App: \[https://seizureaura-ahnwdcwpa3mda9nt6zesfa.streamlit.app/]


