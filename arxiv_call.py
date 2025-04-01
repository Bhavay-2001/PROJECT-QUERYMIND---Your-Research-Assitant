import time
from urllib.error import HTTPError
import os

import arxiv
from tqdm import tqdm
import streamlit as st
from index import create_index

import config

# Function to Fetch Papers from ArXiv
def fetch_papers(query: str, max_results: int) -> list:
    """Fetches research paper metadata from ArXiv based on a search query."""
    search = arxiv.Search(
        query=query,
        max_results=int(max_results),
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "authors": ", ".join([author.name for author in result.authors]),
            "summary": result.summary,
            "pdf_url": result.pdf_url
        })
    return papers
               
def download_paper_by_title_and_index(title: str) -> str:
    st.write(f"Searching for the paper: {title}")
    
    search = arxiv.Search(
        query=f"ti:\"{title}\"",
        max_results=1,  # Fetch only the most relevant paper
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    search_results = list(arxiv.Client().results(search))
    
    if not search_results:
        return "No matching paper found on arXiv. Please check the title."
    
    result = search_results[0]  # Get the first (best match) result
    
    dir_path = os.path.join(config.DOWNLOAD_DIR_PATH, title.replace(' ', '_'))
    os.makedirs(dir_path, exist_ok=True)
    
    while True:
        try:
            result.download_pdf(dirpath=dir_path)
            break
        except FileNotFoundError:
            return "Error: File not found."
        except HTTPError:
            return "Error: HTTP request failed."
        except ConnectionResetError:
            time.sleep(5)
    
    st.write(f"Paper '{title}' downloaded successfully.")
    return create_index(dir_path)


def index_uploaded_paper(uploaded_file) -> str:
    """Indexes a user-uploaded PDF file."""
    dir_path = os.path.join(config.DOWNLOAD_DIR_PATH, "uploaded_papers")
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return create_index(dir_path)