# 🧠 AI Document Processing Suite

An AI-powered pipeline for automated document classification, OCR-based data extraction, and intelligent retrieval using Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG). Designed to streamline the analysis of mortgage and financial documents using open-source tools like PyMuPDF, Tesseract, and LlamaIndex.

---

## 🚀 Features

- 🔍 **OCR + Data Extraction**: Extracts structured data from scanned PDFs using PyMuPDF and Tesseract.
- 🧾 **Document Classification**: Classifies financial documents based on layout and content.
- 🧠 **Retrieval-Augmented Generation (RAG)**: Combines LLMs with LlamaIndex to answer user queries using document context.
- 📊 **Performance Evaluation**: Benchmarked multiple open-source AI models for speed, accuracy, and deployment feasibility.
- 🛠️ **Python-Based Pipeline**: Modular and easily extensible architecture.

---

## 🧰 Tech Stack

- **Python**
- **PyMuPDF** – for text extraction and layout analysis
- **Tesseract OCR** – for image-based text recognition
- **LlamaIndex** – for intelligent retrieval and RAG
- **LangChain** (optional) – for LLM orchestration
- **OpenAI / Cohere / HuggingFace Transformers** – for question answering and summarization

---

## 📂 Project Structure


---

## 🧪 Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-doc-processing-suite.git
cd ai-doc-processing-suite
pip install -r requirements.txt
python src/ocr/extract_text.py --input data/sample_loan.pdf
python src/retrieval/query_docs.py --query "What is the interest rate?"
