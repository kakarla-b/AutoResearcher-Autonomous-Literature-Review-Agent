# AutoResearcher: Autonomous Literature Review Agent (OpenAI + SerpAPI)

AutoResearcher is a command-line tool that automates literature reviews. It fetches academic papers from Google Scholar using SerpAPI, summarizes them with OpenAI and generates structured summaries of key findings, contributions, and methods.

## Features

- Searches for academic papers using SerpAPI + Google Scholar
- Loads and chunks content using LangChain loaders
- Summarizes each paper using OpenAI GPT with structured prompts
- Uses FAISS for vector storage (future expansion possible)
- Clean CLI with topic input and bullet-style summaries
  
## Tech Stack
- Python
- OpenAI GPT-4 / GPT-3.5 (via `openai` API)
- SerpAPI (Google Scholar interface)
- LangChain
- FAISS (Vector storage for document indexing)

## Prerequisites

You will need API keys for:
- [OpenAI](https://platform.openai.com/account/api-keys)
- [SerpAPI](https://serpapi.com/manage-api-key)


## Example Usage
$ python auto_researcher.py
Enter your research topic: generative AI for document summarization
