# Updated-Langchain

A comprehensive collection of LangChain examples and implementations showcasing various AI/ML capabilities including RAG (Retrieval Augmented Generation), chatbots, agents, and integrations with multiple LLM providers.

## 📋 Table of Contents
  
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Examples](#examples)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project demonstrates practical implementations of LangChain for building AI-powered applications. It includes examples of:

- **RAG (Retrieval Augmented Generation)** - Building intelligent Q&A systems with document retrieval
- **Chatbots** - Interactive conversational interfaces using OpenAI and Ollama
- **Agents** - Autonomous AI agents capable of complex reasoning
- **API Integration** - FastAPI servers with LangChain
- **Multiple LLM Providers** - OpenAI, Groq, Hugging Face, and Ollama integrations
- **Vector Stores** - FAISS, ChromaDB, and ObjectBox implementations

## ✨ Features

- 🤖 **Multiple LLM Providers**: Support for OpenAI, Groq, Hugging Face, and Ollama
- 📚 **RAG Implementation**: Document-based Q&A systems with vector embeddings
- 💬 **Interactive Chatbots**: Streamlit-based chat interfaces
- 🔗 **API Integration**: FastAPI servers with LangServe
- 🧠 **AI Agents**: Autonomous agents for complex tasks
- 📊 **Vector Databases**: FAISS, ChromaDB, and ObjectBox support
- 🔍 **Document Processing**: PDF, web scraping, and text processing capabilities

## 📁 Project Structure

```
Updated-Langchain/
├── agents/              # AI agent implementations
│   └── agents.ipynb
├── api/                 # FastAPI server examples
│   ├── app.py          # LangServe API server
│   └── client.py       # API client examples
├── chain/              # LangChain chain examples
│   ├── attention.pdf
│   └── retriever.ipynb
├── chatbot/            # Chatbot implementations
│   ├── app.py         # OpenAI chatbot with Streamlit
│   └── localama.py    # Ollama (Llama2) chatbot
├── groq/               # Groq API integrations
│   ├── app.py         # Groq RAG with web scraping
│   ├── groq.ipynb
│   └── llama3.py     # Llama3 with PDF document processing
├── huggingface/        # Hugging Face integrations
│   ├── huggingface.ipynb
│   └── us_census/     # Sample PDF documents
├── objectbox/          # ObjectBox vector store examples
│   ├── app.py
│   └── us_census/     # Sample PDF documents
├── openai/             # OpenAI-specific implementations
│   └── GPT4o_Lanchain_RAG.ipynb
├── rag/                # RAG examples and resources
│   ├── attention.pdf
│   ├── simplerag.ipynb
│   └── speech.txt
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/krishnaik06/Updated-Langchain.git
cd Updated-Langchain
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file in the root directory with the following API keys:

```env
# OpenAI API Key (required for OpenAI examples)
OPENAI_API_KEY=your_openai_api_key_here

# Groq API Key (required for Groq examples)
GROQ_API_KEY=your_groq_api_key_here

# LangChain API Key (optional, for LangSmith tracking)
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
```

### Getting API Keys

- **OpenAI**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Groq**: Get your API key from [Groq Console](https://console.groq.com/)
- **LangChain/LangSmith**: Get your API key from [LangSmith](https://smith.langchain.com/)

## 📖 Usage

### 1. Chatbot with OpenAI

Run the OpenAI-powered chatbot:

```bash
cd chatbot
streamlit run app.py
```

Access the application at `http://localhost:8501`

### 2. Chatbot with Ollama (Local)

First, ensure Ollama is installed and Llama2 model is downloaded:

```bash
# Install Ollama from https://ollama.ai
ollama pull llama2
```

Then run the local chatbot:

```bash
cd chatbot
streamlit run localama.py
```

### 3. Groq RAG Application

Run the Groq-powered RAG application with web scraping:

```bash
cd groq
streamlit run app.py
```

### 4. Groq with PDF Documents (Llama3)

Place your PDF files in the `groq/us_census/` directory, then run:

```bash
cd groq
streamlit run llama3.py
```

### 5. ObjectBox Vector Store

Run the ObjectBox vector store example:

```bash
cd objectbox
streamlit run app.py
```

Place your PDF files in the `objectbox/us_census/` directory before running.

### 6. FastAPI Server

Start the LangServe API server:

```bash
cd api
python app.py
```

The server will start at `http://localhost:8000`

Available endpoints:
- `/openai` - OpenAI chat endpoint
- `/essay` - Essay generation endpoint
- `/poem` - Poem generation endpoint (using Ollama)

### 7. Jupyter Notebooks

Explore the Jupyter notebooks for interactive learning:

```bash
jupyter notebook
```

Notable notebooks:
- `agents/agents.ipynb` - AI agents examples
- `chain/retriever.ipynb` - Chain and retriever examples
- `groq/groq.ipynb` - Groq integration examples
- `openai/GPT4o_Lanchain_RAG.ipynb` - GPT-4o RAG implementation
- `rag/simplerag.ipynb` - Simple RAG examples
- `huggingface/huggingface.ipynb` - Hugging Face integration

## 💡 Examples

### Example 1: Simple Chatbot Query

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

chain = prompt | llm
response = chain.invoke({"question": "What is LangChain?"})
print(response.content)
```

### Example 2: RAG with Document Retrieval

```python
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load and embed documents
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Create retrieval chain
llm = ChatGroq(model_name="mixtral-8x7b-32768")
retriever = vectorstore.as_retriever()
chain = create_retrieval_chain(retriever, document_chain)

# Query
response = chain.invoke({"input": "Your question here"})
print(response['answer'])
```

## 🛠️ Technologies Used

- **LangChain** - Framework for building LLM applications
- **OpenAI** - GPT models integration
- **Groq** - High-performance LLM inference
- **Ollama** - Local LLM deployment
- **Streamlit** - Interactive web applications
- **FastAPI** - Modern Python web framework
- **LangServe** - LangChain API server
- **FAISS** - Vector similarity search
- **ChromaDB** - Vector database
- **ObjectBox** - Embedded vector database
- **Hugging Face** - Transformers and models
- **PyPDF2** - PDF processing
- **BeautifulSoup4** - Web scraping

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for the amazing framework
- [Krish Naik](https://github.com/krishnaik06) for the original project inspiration
- All contributors and the open-source community

## 📧 Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

---

**Note**: Make sure to keep your API keys secure and never commit them to version control. Always use environment variables or `.env` files for configuration.
