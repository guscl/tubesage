# 🧙‍♂️ TubeSage

TubeSage is an LLM-based app that understands YouTube videos and answers questions about them.

![TubeSage](images/system.png)

## Project Architecture Overview

1. **YouTube Video Transcription Tool**: Converts video content into text.
2. **Text Embedding**: Uses the Ollama server with Llama3.1 to embed the video transcription.
3. **Vector Storage**: Stores the embedded vectors in the Chroma vector database.
4. **Q&A System**: Utilizes LangChain with a custom prompt and RAG injection from the Chroma DB to answer questions about the video.
5. **Web Interface**: Provides a Streamlit chat UI.
6. **Deployment**: Dockerizes all components (TubeSage API, Ollama server for the LLM, and Streamlit UI) and runs them with Docker Compose. Use the provided Dockerfiles for deployment.

## Getting Started

To get started with TubeSage, follow these steps:

### Prerequisites

- **NVIDIA GPU**: Required for LLM inference with Ollama.
- **Docker & Docker Compose**: Ensure Docker and Docker Compose are installed on your system.

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/guscl/tubesage.git
   cd tubesage
   ```

2. **Install Dependencies**

   Ensure you have the NVIDIA toolkit installed for CUDA support.
   - **NVIDIA Toolkit**: [NVIDIA Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation).

3. **Run Docker Compose**

   Use Docker Compose to start all services:

   ```bash
   docker compose up
   ```

4. **Access the Application**

   Open your web browser and navigate to `http://localhost:8501` to interact with the TubeSage interface.

## Usage

1. **Transcribe a Video**

   Enter a YouTube video URL.

2. **Ask Questions**

   Type questions in the textbox at the end of the page. TubeSage will provide answers based on the transcribed and embedded text.

## Contributing

Feel free to submit issues or pull requests. Contributions are welcome!

## Contact

For questions, please contact [correia.gustavol@gmail.com](mailto:correia.gustavol@gmail.com) or open an issue on the [GitHub repository](https://github.com/guscl/tubesage/issues).

## Areas for Improvement

1. **LangChain**: LangChain API is not very well organized; Future experiments with LlamaIndex are planned.
2. **YouTube Transcription API**: This API is a wrapper over an undocumented API, which may become unreliable.
3. **Ollama**: Ollama currently does not feel production-ready. Future experimentation with VLLM is planned.
