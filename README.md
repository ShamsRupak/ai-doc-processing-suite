# <div align="center">🧠 AI Document Processing Suite</div>

<div align="center">
  
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=30&pause=1000&color=2E86AB&center=true&vCenter=true&width=600&lines=Extract+%E2%9E%A1%EF%B8%8F+Classify+%E2%9E%A1%EF%B8%8F+Retrieve;Powered+by+RAG+%2B+Open-Source+AI;Financial+Document+Intelligence" alt="Typing SVG" />
  
  <br/>
  
  [![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
  [![LangChain](https://img.shields.io/badge/🦜_LangChain-0.1.0+-green?style=for-the-badge)](https://langchain.com)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
  
  <br/>
  
  [![Stars](https://img.shields.io/github/stars/ShamsRupak/ai-doc-processing-suite?style=social)](https://github.com/ShamsRupak/ai-doc-processing-suite/stargazers)
  [![Forks](https://img.shields.io/github/forks/ShamsRupak/ai-doc-processing-suite?style=social)](https://github.com/ShamsRupak/ai-doc-processing-suite/network/members)
  [![Issues](https://img.shields.io/github/issues/ShamsRupak/ai-doc-processing-suite?style=social)](https://github.com/ShamsRupak/ai-doc-processing-suite/issues)
  
  <br/>
  
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700">
  
  <h3>🚀 Transform Your Documents into Intelligent, Queryable Knowledge</h3>
  
</div>

---

## 📋 Table of Contents

<details open>
<summary><b>Click to Navigate</b></summary>

- [✨ Features](#-features)
- [🎯 Use Cases](#-use-cases)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [📊 Performance](#-performance)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [💻 Usage Examples](#-usage-examples)
- [🔧 Configuration](#-configuration)
- [📈 Benchmarks](#-benchmarks)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

</details>

---

## ✨ Features

<div align="center">
  
| Feature | Description | Status |
|---------|-------------|--------|
| 🔍 **Smart OCR** | Extract text from scanned PDFs with 98%+ accuracy | ✅ Ready |
| 📑 **Auto Classification** | Categorize documents using AI-powered analysis | ✅ Ready |
| 🧠 **RAG Pipeline** | Answer questions using document context | ✅ Ready |
| 📊 **Multi-Model Support** | Compare TinyLlama, Phi-2, Mistral performance | ✅ Ready |
| 🚄 **Optimized Processing** | GPU-accelerated with batch processing | ✅ Ready |
| 🔐 **Secure Handling** | PII detection and redaction capabilities | 🚧 Coming |

</div>

### 🎬 Demo

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/225813708-98b745f2-7d22-48cf-9150-083f1b00d6c9.gif" width="600">
</div>

---

## 🎯 Use Cases

<details>
<summary><b>🏦 Financial Services</b></summary>

- **Mortgage Processing**: Extract key terms from loan documents
- **Contract Analysis**: Identify important clauses and conditions
- **Compliance Checking**: Ensure documents meet regulatory requirements

</details>

<details>
<summary><b>📋 Legal Operations</b></summary>

- **Document Discovery**: Search through large document sets
- **Contract Review**: Extract and analyze key terms
- **Due Diligence**: Automated document verification

</details>

<details>
<summary><b>🏢 Enterprise Solutions</b></summary>

- **Invoice Processing**: Extract line items and totals
- **Report Generation**: Summarize lengthy documents
- **Knowledge Management**: Build searchable document repositories

</details>

---

## 🏗️ Architecture

```mermaid
graph LR
    A[📄 PDF Input] --> B[🔍 OCR Engine]
    B --> C[📑 Classifier]
    C --> D[🧩 Chunking]
    D --> E[🔢 Embeddings]
    E --> F[💾 Vector Store]
    F --> G[🤖 LLM + RAG]
    G --> H[💬 Answer]
    
    style A fill:#e1f5fe
    style H fill:#c8e6c9
---

## 🚀 Quick Start

### Prerequisites

<div align="center">

| Requirement | Version |
|-------------|---------|
| Python | 3.8+ |
| CUDA | 11.8+ (for GPU) |
| RAM | 8GB minimum |
| Storage | 10GB free |

</div>

### 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/ShamsRupak/ai-doc-processing-suite.git
cd ai-doc-processing-suite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py

from doc_processor import DocumentPipeline

# Initialize pipeline
pipeline = DocumentPipeline(model="tinyllama")

# Process document
result = pipeline.process("data/sample_loan.pdf")

# Query the document
answer = pipeline.query("What is the interest rate?")
print(f"Answer: {answer}")
