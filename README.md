# 🚗 Used Car Price Co-Pilot

**Your AI sidekick for navigating the wonderfully chaotic world of used car pricing in Pakistan.**

Tired of Uncle Jameel's "expert" opinion on your car's value? Baffled by online listings that range from "practically giving it away" to "are you selling a car or a small planet?" Fear not! The Used Car Price Co-Pilot is here to be the calm, data-driven voice of reason in your head.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green?style=for-the-badge&logo=langchain)](https://www.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini_Pro-purple?style=for-the-badge&logo=google-gemini)](https://ai.google.dev/)

## 🚀 Live Demo

Why read about it when you can try it? Click the link below to chat with the Co-Pilot right now!

### [➡️ Click Here to Launch the Live App ⬅️](https://usedcarpricechatbot.streamlit.app/)

Watch as a simple, everyday sentence is transformed into a detailed, justified price analysis. It's like magic, but with more Python and less pulling a rabbit out of a hat.

![Demo GIF](https://i.imgur.com/REPLACE_WITH_YOUR_GIF.gif) 
*Note: Replace the placeholder link above with a real GIF of your application for maximum effect!*

---

## 🤔 What Does This Thing Actually Do?

Glad you asked! This isn't just another boring price calculator. It's a **conversational pricing engine**. It combines the raw number-crunching power of a machine learning model with the contextual understanding of a state-of-the-art Large Language Model (LLM).

Here's the breakdown of its core duties:

*   **💬 Conversational Q&A:** You can chat with it! Ask about specific features, market trends, or what "unregistered" really means for the price. It has access to a knowledge base to answer your general questions.
*   **🕵️‍♂️ Intelligent Feature Extraction:** Throw a wall of text at it. The Co-Pilot acts as a digital detective, meticulously picking out the vital details: `brand`, `model`, `year`, `mileage`, `color`, and even fancy add-ons like `Sun Roof` or `Heated Seats`.
*   **🧠 Data-Driven Price Prediction:** The number-crunching heart of the operation is a `RandomForestRegressor` model, trained on a dataset of used car prices. It takes the extracted features and generates a baseline algorithmic price.
*   **🎩 Expert Synthesis & Justification:** This is the real magic. The LLM takes the raw number from the ML model, considers all the provided details, and synthesizes a final, human-readable **price range** and a **justification**. It's the difference between a calculator and a consultant.

---

## ⚙️ How It Works (The Slightly Comedic, Gory Details)

Our Co-Pilot operates on a sophisticated, multi-chain "attack" strategy to wrestle a price out of thin air. It's a beautiful, slightly over-engineered symphony of AI.

1.  **The Greeting Committee (RAG Chain):**
    When you first type a message, it passes through a **Retrieval-Augmented Generation (RAG)** chain. This chain has two goals:
    *   Remember your conversation history so you don't have to repeat yourself.
    *   Consult its knowledge base (`my_database.txt`) to answer any general questions you might have. It's the friendly, knowledgeable face of the operation.

2.  **The Interrogator (Feature Extraction Chain):**
    Simultaneously, your input is sent to the **Feature Extraction Chain**. This is a stern, no-nonsense LLM prompt that has one job: "Extract the vehicle features and give them to me as a clean JSON. No excuses." It's ruthlessly efficient.

3.  **The Moment of Truth (The Prediction):**
    *   If the "Interrogator" fails to find the `brand`, `car`, and `model_year`, the process stops, and you get a friendly conversational answer.
    *   If it *succeeds*, the extracted JSON is handed off to our `predict_price` function. This function preps the data into a format our `RandomForestRegressor` model can understand and... **BAM!** We get a raw price estimate.

4.  **The Grand Finale (Price Synthesis Chain):**
    The raw price is... well, raw. The final chain takes this number, along with all the features the Interrogator found, and performs the final synthesis. It's prompted to act as a "top-tier used car pricing expert," using the number as a reference to create a realistic market range and a snappy justification for its reasoning.

This entire process results in a response that is both data-driven and contextually aware. All in the time it takes to sip your chai.

---

## 🛠️ How to Run This Marvel of Engineering

Want to get this running locally? Of course you do.

1.  **Clone the Sacred Texts:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Set Up Your Environment:**
    A virtual environment is a pro-move. Don't be an amateur.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Provide the Secret Sauce (API Key):**
    The Co-Pilot's brain (Google's Gemini model) requires an API key to function.
    *   Create a file named `.env` in the root directory.
    *   Inside `.env`, add the following line:
        ```
        GOOGLE_API_KEY="your_super_secret_api_key_here"
        ```
    *   Without this, the AI will remain dormant, dreaming of electric sheep.

4.  **Ensure the Brains are in Place:**
    Make sure you have the pre-trained model and data files in their respective directories:
    *   `Models/Random_Forest_Regressor_Model.pkl`
    *   `Data/my_database.txt`

5.  **LIFTOFF!**
    Run the Streamlit application from your terminal:
    ```bash
    streamlit run app.py
    ```
    Your browser should open a new tab with the Co-Pilot, ready for duty.

---

### **Disclaimer**
This is a sophisticated demo project. The price estimates are based on a specific model and dataset and are enhanced by an LLM. While it's impressively accurate, don't base life-altering financial decisions solely on our AI's word, no matter how confident it sounds. Always consult multiple sources. And maybe Uncle Jameel. Just to be polite.
