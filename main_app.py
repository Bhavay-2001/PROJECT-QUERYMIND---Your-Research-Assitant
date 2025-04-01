import streamlit as st
import tempfile
from gtts import gTTS
import uuid

from arxiv_call import download_paper_by_title_and_index, index_uploaded_paper, fetch_papers
from model import ArxivModel

tab1, tab2 = st.tabs(["Search ARXIV Papers", "Chat with Papers"])

with tab1:
    st.header("Search ARXIV Papers")
    search_input = st.text_input("Search query")
    num_papers_input = st.number_input("Number of papers", min_value=1, value=5, step=1)
    result_placeholder = st.empty()

    if st.button("Search"):
        if search_input:
            papers_info = fetch_papers(search_input, num_papers_input)
            result_placeholder.empty()
            if papers_info:
                st.subheader("Search Results:")
                for i, paper in enumerate(papers_info, start=1):
                    with st.expander(f"**{i}. {paper['title']}**"):
                        st.write(f"**Authors:** {paper['authors']}")
                        st.write(f"**Summary:** {paper['summary']}")
                        st.write(f"[Read Paper]({paper['pdf_url']})")
            else:
                st.warning("No papers found. Try a different query.")
        else:
            st.warning("Please enter a search query.")

with tab2:
    st.header("Talk to the Papers")

    if st.button("Clear Chat", key="clear_chat_button"):
        st.session_state.clear()
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "COLLECTION_NAME" not in st.session_state:
        st.session_state.COLLECTION_NAME = None
    if "arxiv_model" not in st.session_state:
        st.session_state.arxiv_model = None
    if "llm_chain" not in st.session_state:
        st.session_state.llm_chain = None
    if "session_config" not in st.session_state:
        st.session_state.session_config = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                try:
                    tts = gTTS(message["content"])
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                        tts.save(tmp_file.name)
                        tmp_file.seek(0)
                        st.audio(tmp_file.read(), format="audio/mp3")
                except Exception as e:
                    st.error("Text-to-speech failed.")
                    st.error(str(e))

    paper_title = st.text_input("Enter the title of the paper to fetch from ArXiv:")
    uploaded_file = st.file_uploader("Or upload a research paper (PDF):", type=["pdf"])

    if st.button("Index Paper"):
        if paper_title:
            st.session_state.indexed_paper = paper_title
            with st.spinner("Fetching and indexing paper..."):
                st.session_state.COLLECTION_NAME = paper_title
                result = download_paper_by_title_and_index(paper_title)
                st.success(result)
        elif uploaded_file:
            st.session_state.indexed_paper = uploaded_file.name
            with st.spinner("Indexing uploaded paper..."):
                st.session_state.COLLECTION_NAME = uploaded_file.name[:-4]
                result = index_uploaded_paper(uploaded_file)
                st.success(result)
        else:
            st.warning("Please enter a paper title or upload a PDF.")

        # Reset model to ensure fresh load after new paper is indexed
        st.session_state.arxiv_model = ArxivModel(
            session_id=st.session_state.session_id,
            collection_name=st.session_state.COLLECTION_NAME
        )
        st.session_state.llm_chain, st.session_state.session_config = st.session_state.arxiv_model.get_model()

    def process_chat(prompt):
        if not st.session_state.arxiv_model:
            st.session_state.arxiv_model = ArxivModel(
                session_id=st.session_state.session_id,
                collection_name=st.session_state.COLLECTION_NAME
            )
            st.session_state.llm_chain, st.session_state.session_config = st.session_state.arxiv_model.get_model()

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            response = st.session_state.llm_chain.invoke(
                {"input": prompt},
                config=st.session_state.session_config
            )["answer"]

        with st.chat_message("assistant"):
            st.markdown(response)
            try:
                tts = gTTS(response)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tts.save(tmp_file.name)
                    tmp_file.seek(0)
                    st.audio(tmp_file.read(), format="audio/mp3")
            except Exception as e:
                st.error("Text-to-speech failed.")
                st.error(str(e))

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})

    if user_query := st.chat_input("Ask a question about the papers..."):
        process_chat(user_query)

    if st.button("Clear Recent Chat"):
        st.session_state.messages = []
        st.session_state.arxiv_model = None
        st.session_state.llm_chain = None
        st.session_state.session_config = None
