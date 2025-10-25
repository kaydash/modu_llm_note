# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a collection of educational Jupyter notebooks focused on AI development with Python, LangChain, and RAG (Retrieval-Augmented Generation) systems. The notebooks are structured as a learning curriculum organized by weeks:

- **Week 1**: Python fundamentals, OpenAI API, LangChain basics, LangSmith
- **Week 2**: RAG implementation, document processing, embeddings, vector stores
- **Week 3**: Advanced prompt engineering, chat systems, practical applications

## Development Environment

### Jupyter Notebook Environment
- All code is developed in Jupyter notebooks (.ipynb files)
- Execute cells sequentially for proper workflow
- Use IPython features like `display()` for rich output formatting
- Notebooks contain both educational content and practical implementations

### Environment Setup
- Create `.env` file for API keys: `OPENAI_API_KEY=your_key_here`
- Load environment variables using `python-dotenv`: `load_dotenv()`
- All notebooks expect OpenAI API access for LLM functionality

## Key Dependencies

### Core Libraries
```bash
# LangChain ecosystem
langchain
langchain-community
langchain-openai
langchain-chroma
langchain_text_splitters

# AI/ML libraries
openai
faiss-cpu

# Data processing
beautifulsoup4
python-dotenv

# UI framework
gradio

# Image processing
pillow
httpx

# Jupyter environment
ipykernel
```

### Package Management
- Uses `uv` package manager where mentioned (recommended)
- Fallback to `pip` for standard installations
- Virtual environment setup: `uv venv --python=3.12`

## Architecture Patterns

### RAG Pipeline Structure
1. **Document Loading**: WebBaseLoader for web content, various loaders for different formats
2. **Text Splitting**: CharacterTextSplitter with configurable chunk sizes (typically 1000 chars with 200 overlap)
3. **Embeddings**: OpenAI text-embedding-3-small model for vector representations
4. **Vector Storage**: Chroma vector database for similarity search
5. **Retrieval**: Similarity-based retrieval with configurable k values
6. **Generation**: ChatOpenAI models (gpt-4.1-mini, gpt-4o-mini) for response generation

### Chain Composition
- Uses LangChain's `create_retrieval_chain()` and `create_stuff_documents_chain()`
- Prompt templates with system and human message roles
- Structured responses with context and answer separation

### UI Implementation
- Gradio ChatInterface for interactive chatbot interfaces
- Async/await patterns for responsive UI
- Integration with RAG chains for knowledge-based responses

## Common Development Tasks

### Working with Notebooks
- Run cells in sequence to maintain state
- Check for environment variable loading at the start
- Use `print()` statements for debugging outputs
- Handle large outputs with pagination or truncation

### Testing RAG Systems
- Test document loading with sample URLs or files
- Verify text splitting produces appropriate chunk sizes
- Check embedding generation and vector storage
- Validate retrieval with similarity searches
- Test end-to-end QA chain functionality

### API Integration
- Always check API key availability before making calls
- Handle rate limiting and error responses gracefully
- Monitor token usage with response.usage attributes
- Use appropriate model selection based on task complexity

## File Organization

### Notebook Naming Convention
- Format: `PRJ01_W{week}_{sequence}_{topic}.ipynb`
- Week 1: Fundamentals (Python, OpenAI, LangChain basics)
- Week 2: RAG implementation (tokenizing, embeddings, vector stores)
- Week 3: Advanced applications (prompt engineering, chatbots)

### Content Structure
Each notebook typically contains:
- Markdown cells with learning objectives and explanations
- Code cells with practical implementations
- Example outputs and visualizations
- Practice exercises at the end

## Technical Considerations

### Memory Management
- Jupyter notebooks can accumulate memory usage over time
- Restart kernels periodically for long-running sessions
- Be mindful of large vector stores and embedding collections

### Error Handling
- Handle OpenAI API errors (rate limits, authentication)
- Manage vector store connection issues
- Graceful degradation when external resources are unavailable

### Performance Optimization
- Use appropriate chunk sizes for document splitting
- Configure retrieval parameters based on use case
- Monitor embedding model costs and token usage

## Troubleshooting

### Common Issues
- LangSmith authentication warnings can be ignored if not using LangSmith
- Vector store persistence may require explicit configuration
- Large notebook files may need memory management

### Debugging Strategies
- Check cell execution order for state dependencies
- Verify environment variable loading
- Test individual components before full chain integration
- Use smaller datasets for initial development and testing