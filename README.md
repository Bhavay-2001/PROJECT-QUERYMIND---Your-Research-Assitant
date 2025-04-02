# QUERYMIND---Your-Research-Assitant


**QueryMind** is an AI-powered virtual assistant that helps users simplify academic research. It allows users to **search for papers from ArXiv**, **upload and interact with research papers**, and **ask natural language questions** — all through a friendly web interface built with Streamlit. The assistant uses **LLMs**, **RAG (Retrieval-Augmented Generation)**, **vector databases**, and **Text-to-Speech (TTS)** to make exploring research easier and more interactive.

---

## 🚀 Features

- 🔍 **Search Papers from ArXiv** by entering a research topic
- 📄 **Upload/Index Research Papers** (PDFs or by title)
- 💬 **Chat with Papers** using intelligent Q&A based on paper content
- 🧠 **Chat History Memory** to handle multi-turn questions
- 🗣️ **Text-to-Speech Output** for listening to responses
- ⚡ Uses **LangChain**, **Qdrant**, **Groq LLMs**, and **Streamlit**

---

## 🛠️ How to Run This Project

Follow the steps below to set up and run the app locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Bhavay-2001/PROJECT-QUERYMIND---Your-Research-Assitant.git
cd PROJECT-QUERYMIND---Your-Research-Assitant
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
conda create -n querymind python=3.10
conda activate querymind
```

### 3. Install Dependencies

Make sure you have `pip` updated and install all requirements:

```bash
pip install -r requirements.txt
```

### 4. Add API Keys

Create a `.env` file in the root directory and add the following keys:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
QDRANT_API_KEY=your_qdrant_api_key
```

> You can sign up and get API keys from:
> - [Google Cloud](https://console.cloud.google.com/)
> - [Groq Cloud](https://console.groq.com/)
> - [Hugging Face](https://huggingface.co/settings/tokens)
> - [Qdrant Cloud](https://cloud.qdrant.io/)

### 5. Run the App

```bash
streamlit run main_app.py
```

The app will launch in your browser at: `http://localhost:8501`

---

## 📦 Project Structure

```text
PROJECT-QUERYMIND
├── main_app.py           # Streamlit UI and app logic
├── model.py              # RAG pipeline and chatbot logic
├── arxiv_call.py         # Handles ArXiv paper fetching and indexing
├── index.py              # PDF chunking, embeddings, and Qdrant vector DB setup
├── config.py             # All configuration and API key management
├── arxiv_pdfs/           # Folder to store downloaded/uploaded PDFs
├── .env                  # Environment variables (not pushed to Git)
└── requirements.txt      # All Python package dependencies
```

---

## 📚 Tech Stack

| Component         | Technology Used                          |
|------------------|-------------------------------------------|
| UI               | Streamlit                                 |
| Language Model   | Gemma2-9b-it via Groq                     |
| Vector DB        | Qdrant (Cloud)                            |
| Embeddings       | HuggingFace (mpnet-base-v2)               |
| TTS              | Google Text-to-Speech (`gTTS`)            |
| Retrieval        | LangChain + Conversational RAG            |
| Paper Fetching   | ArXiv API                                 |

---

## 📌 Example Workflow

1. **Search for “LLMs”** → Get top research papers from ArXiv.
2. **Upload or Index a Paper** → Automatically chunked & stored in Qdrant.
3. **Ask Questions** like:
   - *"Give me 5 key contributions."*
   - *"Explain the second contribution in detail."*
4. **Get audio output** if needed.

---

## ✅ Deliverables

- Interactive research paper chatbot
- Streamlit web interface
- ArXiv search and PDF upload support
- Vector database storage and retrieval
- Natural language Q&A with conversation memory
- Optional audio responses via TTS

---

## 🤝 Contributions

This is a student research project built for learning purposes.  
Feel free to fork, contribute or raise issues!

---

## 🧑‍💻 Authors

- [Bhavay Malhotra](https://github.com/Bhavay-2001)
- [Ajay Sam Victor](https://github.com/AJAYSAM02)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
