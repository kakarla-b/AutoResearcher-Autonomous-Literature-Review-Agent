### AutoResearcher: Autonomous Literature Review Agent

# This is the main Python script that powers the AutoResearcher agent.
# It takes a research query, uses SerpAPI to fetch academic links,
# summarizes them with LLM, and outputs a clean literature overview.

import os
import time
import requests
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Initialize LLM
llm = OpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY)

# Prompt Template for summarizing papers
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["content"],
    template="""
    You are a helpful academic assistant. Summarize the key contributions, methodology,
    and findings of the following research paper in bullet points:

    {content}
    """
)
summarizer = LLMChain(llm=llm, prompt=SUMMARY_PROMPT)

# --- STEP 1: Get Research Links from Google Scholar ---
def get_research_links(query: str, max_results: int = 5):
    url = "https://serpapi.com/search.json"
    params = {
        "q": query,
        "engine": "google_scholar",
        "api_key": SERPAPI_KEY,
    }
    response = requests.get(url, params=params)
    results = response.json()
    links = []
    for item in results.get("organic_results", [])[:max_results]:
        if "link" in item:
            links.append(item["link"])
    return links

# --- STEP 2: Load, Chunk, and Embed Documents ---
def load_and_embed_docs(urls):
    loader = WebBaseLoader(urls)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(split_docs, embeddings)
    return split_docs, vectordb

# --- STEP 3: Summarize Each Paper ---
def summarize_documents(documents):
    summaries = []
    for doc in documents:
        try:
            result = summarizer.run(content=doc.page_content[:3000])
            summaries.append(result)
        except Exception as e:
            print("Skipping doc due to error:", e)
    return summaries

# --- Main Flow ---
def run_auto_researcher(topic):
    print(f"\n[+] Searching for papers on: {topic}")
    links = get_research_links(topic)
    print(f"[+] Found {len(links)} paper links.")

    print("[+] Loading and embedding documents...")
    docs, db = load_and_embed_docs(links)

    print("[+] Generating summaries...")
    summaries = summarize_documents(docs)

    print("\n========= LITERATURE REVIEW =========\n")
    for i, summary in enumerate(summaries):
        print(f"Paper {i+1} Summary:\n{summary}\n")

if __name__ == "__main__":
    research_topic = input("Enter your research topic: ")
    run_auto_researcher(research_topic)
