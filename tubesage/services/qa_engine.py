import logging
from tubesage.clients.llm_client import LLMClient
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class QAEngine:
    def invoke(self, input: str):
        pass


class LangChainQA(QAEngine):
    def __init__(self, llm_client: LLMClient, vector_retriever: VectorStoreRetriever):
        try:
            prompt = ChatPromptTemplate.from_template(
                """
                Please answer the following question based on the provided context and your internal knowledge. 
                Prioritize the provided context, and if you are unsure of the answer, state that you are not aware of the topic.

                First, analyze the context carefully and identify the key information relevant to the question. 
                Then, consider how this information relates to the question being asked. 
                Finally, provide a clear answer based on your analysis.
                                                    
                Be clear whether or not the provided answer is based on the context or your internal knowledge.
                If the context was relevant, also point out the specific part of the context that led you to the answer.

                <context>
                {context}
                </context>

                Question: {input}
            """
            )

            document_chain = create_stuff_documents_chain(llm_client.get_llm(), prompt)
            self.retrieval_chain = create_retrieval_chain(vector_retriever, document_chain)

            logger.info("Initialized LangChainQA")
        except Exception as e:
            logger.error(f"Failed to initialize LangChainQA: {e}")
            raise

    def invoke(self, input: str):
        try:
            return self.retrieval_chain.invoke(input)["answer"]
        except Exception as e:
            logger.error(f"An error occurred while invoking LangChainQA: {e}")
            raise


if __name__ == "__main__":
    import asyncio

    async def main():
        from tubesage.clients.llm_client import OllamaLLMClient
        from tubesage.services.text_splitter import LangChainSmartTextSplitter
        from tubesage.clients.embbeding_client import OllamaEmbeddingClient
        from tubesage.clients.video_transcript import YoutubeTranscriptClient
        from tubesage.clients.vector_db import ChromaClient

        text_splitter = LangChainSmartTextSplitter(chunk_size=4000)
        embedding_client = OllamaEmbeddingClient("llama3.1")
        llm_client = OllamaLLMClient(model="llama3.1")

        yt_transcript_client = YoutubeTranscriptClient()
        key = "VMj-3S1tku0&t"

        _, full_transcript = yt_transcript_client.get_transcript(key)

        chroma_client = ChromaClient(text_splitter, embedding_client)
        await chroma_client.add_async_texts(full_transcript)
        qa_engine = LangChainQA(llm_client, chroma_client.get_retriever())

        print(qa_engine.invoke({"input": "What is the video about?"}))

    asyncio.run(main())
