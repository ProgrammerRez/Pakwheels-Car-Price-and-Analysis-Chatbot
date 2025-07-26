import os
import json
import joblib
import asyncio
import pandas as pd
import streamlit as st
import uuid

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# === Load .env ===
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Missing GOOGLE_API_KEY in .env file.")
    st.stop()

# === File Paths & Constants ===
MODEL_PATH = "Models/Random_Forest_Regressor_Model.pkl"
KNOWLEDGE_PATH = "Data/my_database.txt"
VECTORSTORE_DIR = "vectorstore/faiss_index"

ALL_FEATURES = [
    'model_year', 'mileage', 'engine_capacity', 'registered_in', 'color', 'brand', 'car', 'body_type', 'engine_type', 'transmission',
    'ABS', 'AM/FM Radio', 'Air Bags', 'Air Conditioning', 'Alloy Rims', 'CD Player',
    'Cassette Player', 'Climate Control', 'CoolBox', 'Cruise Control', 'DVD Player',
    'Front Camera', 'Front Speakers', 'Heated Seats', 'Immobilizer Key', 'Keyless Entry',
    'Navigation System', 'Power Locks', 'Power Mirrors', 'Power Steering', 'Power Windows',
    'Rear AC Vents', 'Rear Camera', 'Rear Seat Entertainment', 'Rear Speakers',
    'Steering Switches', 'Sun Roof', 'USB and Auxillary Cable'
]
ESSENTIAL_FEATURES_FOR_PREDICTION = ['brand', 'car', 'model_year']

# === LLM + Embeddings ===
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)

# === Async Compatibility ===
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# === UI Sidebar (Tutorial) ===
with st.sidebar:
    st.info("### 🚗 How to Use the Price Estimator")
    st.markdown("""
        Welcome! This tool helps you estimate the market price of used cars in Pakistan. For the best results, provide as much detail as you can.
    """)
    st.header("Example Prompts")
    st.code("Suzuki Alto VXR 2021, 45000 km driven, registered in Lahore")
    st.code("2018 Toyota Corolla Altis 1.6, automatic, silver, with cruise control and alloy rims")
    st.code("What's the price for a Honda Civic Oriel 2022 with a sunroof?")
    st.header("Key Features for Accurate Pricing")
    st.subheader("✅ Essential")
    st.markdown("- **Brand** (e.g., Suzuki, Toyota)\n- **Car/Model** (e.g., Alto, Corolla)\n- **Model Year** (e.g., 2019)")
    st.subheader("👍 Recommended")
    st.markdown("- **Mileage** (e.g., 50000 km)\n- **Transmission** (Automatic or Manual)\n- **Variant** (e.g., VXR, Altis 1.6)\n- **Key Features** (e.g., Sunroof, Cruise Control)")

# === Load Models & Vectorstores ===
@st.cache_resource
def load_vectorstore():
    if os.path.exists(VECTORSTORE_DIR):
        return FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    loader = TextLoader(KNOWLEDGE_PATH)
    docs = loader.load()
    chunks = CharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(VECTORSTORE_DIR)
    return vs

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# === LangChain Chains Setup ===

# --- This is the conversational RAG chain ---
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Use the chat history and follow-up question to generate a standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
retriever = load_vectorstore().as_retriever()
history_aware_retriever = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=CONDENSE_PROMPT)
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question using the provided context."),
    MessagesPlaceholder("chat_history"), ("system", "{context}"), ("human", "{input}")
])
doc_chain = create_stuff_documents_chain(llm, QA_PROMPT)
rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
rag_with_history = RunnableWithMessageHistory(rag_chain, get_session_history, input_messages_key="input", history_messages_key="chat_history")


# Feature Extraction Chain
extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", f"Extract the following fields from the user's text into valid JSON: {ALL_FEATURES}. If a value is not present, omit it from the JSON. Return only the JSON object."),
    ("human", "{input}")
])
extraction_chain = extraction_prompt | llm

# Final Price Synthesis Chain
PRICE_SYNTHESIS_PROMPT = ChatPromptTemplate.from_template(
    """You are a top-tier used car pricing expert for the Pakistani market. Your task is to synthesize all available information to determine a final, fair market price range. You have been given a set of vehicle details and an initial price estimate from a data-driven algorithm to use as a reference point.

    VEHICLE DETAILS: {details}
    ALGORITHMIC REFERENCE PRICE: Rs. {ml_price}

    Based on all of this information, determine a final price range and provide a justification. The justification should be brief and insightful.
    Return your final analysis as a valid JSON object with the following keys:
    - "final_price_range": A string representing a confident and realistic market price range (e.g., "Rs. 2,850,000 to Rs. 3,100,000").
    - "justification": A concise explanation for your pricing, mentioning key factors.
    Output only the JSON object.
    """
)
price_synthesis_chain = PRICE_SYNTHESIS_PROMPT | llm

# === Prediction Function ===
def predict_price(info: dict):
    model = load_model()
    model_features = model.feature_names_in_
    prediction_data = {feature: 0 for feature in model_features}
    for key, value in info.items():
        if key in ['model_year', 'mileage', 'engine_capacity']:
            try: prediction_data[key] = float(value)
            except (ValueError, TypeError): prediction_data[key] = 0
        elif key in prediction_data: prediction_data[key] = 1 if value else 0
        else:
            categorical_feature_name = f"{key}_{value}"
            if categorical_feature_name in prediction_data:
                prediction_data[categorical_feature_name] = 1
    df = pd.DataFrame([prediction_data], columns=model_features)
    price = model.predict(df)[0]
    return int(price)

# === Main Streamlit App UI ===
st.title("🚗 Used Car Price Co-Pilot")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Describe a car or ask a question...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Analyzing your request..."):
        session_id = st.session_state.session_id
        
        # Always get the conversational response first.
        rag_response = rag_with_history.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
        conversational_answer = rag_response.get("answer", "I'm not sure how to answer that, but I can help with pricing.")

        # Try to extract features for pricing.
        raw_extraction = extraction_chain.invoke({"input": user_input})
        try:
            cleaned_content = raw_extraction.content.strip().replace("```json", "").replace("```", "").strip()
            info = json.loads(cleaned_content)
        except (json.JSONDecodeError, AttributeError):
            info = {}

        # Check if essential features for pricing were found.
        missing_features = [f for f in ESSENTIAL_FEATURES_FOR_PREDICTION if not info.get(f)]
        
        # If we have enough info, proceed with pricing.
        if not missing_features:
            ml_price = predict_price(info)
            synthesis_response = price_synthesis_chain.invoke({
                "details": json.dumps(info, indent=2), "ml_price": f"{ml_price:,}"
            })
            try:
                expert_analysis = json.loads(synthesis_response.content.strip().replace("```json", "").replace("```", "").strip())
                price_range = expert_analysis.get("final_price_range", f"Around Rs. {ml_price:,}")
                justification = expert_analysis.get("justification", "Could not generate detailed analysis.")

                # Format the detailed pricing reply
                reply = f"### **{price_range}**\n\n"
                reply += f"**Expert Justification:** *{justification}*\n\n---\n\n"
                entered_features_str = ""
                for key, value in info.items():
                    if value: entered_features_str += f"- **{key.replace('_', ' ').title()}:** {value}\n"
                if entered_features_str: reply += "#### ✅ Details Provided:\n" + entered_features_str
                reply += "\n*Other features were assumed to be standard for the model unless specified otherwise.*\n\n"
                
                # Add the conversational answer at the end
                reply += f"---\n**Additional Insight:** {conversational_answer}"

            except (json.JSONDecodeError, AttributeError):
                reply = f"Based on a preliminary analysis, the price is likely around **Rs. {ml_price:,}**. For a more detailed breakdown, please try rephrasing.\n\n---\n**Context:** {conversational_answer}"
        else:
            # If not enough info for pricing, just give the conversational answer.
            reply = conversational_answer

        st.chat_message("assistant").markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})