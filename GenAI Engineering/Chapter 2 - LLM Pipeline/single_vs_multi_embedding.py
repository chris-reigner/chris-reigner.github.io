# LangChain Multi-Vector Retriever Implementation
# Based on LangChain's official documentation and examples

import uuid
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pydantic import BaseModel, Field

# Sample documents for demonstration
SAMPLE_DOCS = [
    Document(
        page_content="""
        Artificial Intelligence (AI) is a broad field of computer science focused on 
        creating systems capable of performing tasks that typically require human 
        intelligence. This includes machine learning, which uses algorithms to learn 
        patterns from data without explicit programming. Deep learning, a subset of 
        machine learning, employs neural networks with multiple layers to model 
        complex relationships in data. Applications span computer vision, natural 
        language processing, robotics, and recommendation systems.
        """,
        metadata={"source": "ai_overview.txt", "topic": "artificial_intelligence"},
    ),
    Document(
        page_content="""
        Vector databases are specialized storage systems designed to handle 
        high-dimensional vector data efficiently. They enable semantic search by 
        computing similarity measures between query vectors and stored document 
        embeddings. Popular vector databases include Pinecone, Weaviate, Chroma, 
        and Qdrant. These systems support operations like nearest neighbor search, 
        filtering, and real-time updates, making them essential for retrieval-augmented 
        generation (RAG) applications and recommendation systems.
        """,
        metadata={"source": "vector_db.txt", "topic": "databases"},
    ),
    Document(
        page_content="""
        Retrieval-Augmented Generation (RAG) combines the power of pre-trained large 
        language models with external knowledge retrieval. The process involves 
        embedding documents into vectors, storing them in vector databases, retrieving 
        relevant context based on user queries, and generating responses that incorporate 
        this retrieved information. This approach significantly reduces hallucination, 
        provides access to up-to-date information, and allows for domain-specific 
        knowledge integration without fine-tuning the entire model.
        """,
        metadata={"source": "rag_systems.txt", "topic": "nlp"},
    ),
]


class LangChainMultiVectorDemo:
    """Demonstrates LangChain's MultiVectorRetriever with different strategies"""

    def __init__(self, openai_api_key: str = None):
        """
        Initialize the demo. In practice, you'd set your OpenAI API key.
        For this demo, we'll simulate the functionality.
        """
        self.embeddings = OpenAIEmbeddings() if openai_api_key else None
        self.llm = ChatOpenAI(model="gpt-4o-mini") if openai_api_key else None
        self.docs = SAMPLE_DOCS

        # For demo purposes without API key
        if not openai_api_key:
            print("Note: This demo requires OpenAI API key for full functionality.")
            print("Showing structure and code examples instead.\n")

    def strategy_1_smaller_chunks(self):
        """Strategy 1: Split documents into smaller chunks, embed chunks, return parent docs"""
        print("STRATEGY 1: SMALLER CHUNKS")
        print("-" * 40)

        # The vectorstore to use to index the child chunks
        vectorstore = Chroma(
            collection_name="full_documents", embedding_function=self.embeddings
        )

        # The storage layer for the parent documents
        store = InMemoryByteStore()
        id_key = "doc_id"

        # The retriever (empty to start)
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
        )

        # Generate unique IDs for parent documents
        doc_ids = [str(uuid.uuid4()) for _ in self.docs]

        # Split documents into smaller chunks
        child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
        sub_docs = []

        for i, doc in enumerate(self.docs):
            _id = doc_ids[i]
            _sub_docs = child_text_splitter.split_documents([doc])
            for _doc in _sub_docs:
                _doc.metadata[id_key] = _id
            sub_docs.extend(_sub_docs)

        print(f"Original documents: {len(self.docs)}")
        print(f"Chunks created: {len(sub_docs)}")

        # Example of how chunks look
        print("\nExample chunks from first document:")
        first_doc_chunks = [
            doc for doc in sub_docs if doc.metadata.get(id_key) == doc_ids[0]
        ]
        for i, chunk in enumerate(first_doc_chunks[:2]):
            print(f"Chunk {i + 1}: {chunk.page_content[:100]}...")

        # Add to retriever (would work with actual embeddings)
        if self.embeddings:
            retriever.vectorstore.add_documents(sub_docs)
            retriever.docstore.mset(list(zip(doc_ids, self.docs)))

            # Example search
            results = retriever.invoke("What is machine learning?")
            print(f"\nSearch results: {len(results)} documents returned")

        return retriever

    def strategy_2_summaries(self):
        """Strategy 2: Create summaries of documents, embed summaries, return full docs"""
        print("\nSTRATEGY 2: DOCUMENT SUMMARIES")
        print("-" * 40)

        # Create summarization chain
        if self.llm:
            chain = (
                {"doc": lambda x: x.page_content}
                | ChatPromptTemplate.from_template(
                    "Summarize the following document:\n\n{doc}"
                )
                | self.llm
                | StrOutputParser()
            )

            # Generate summaries
            summaries = chain.batch(self.docs, {"max_concurrency": 5})
        else:
            # Mock summaries for demo
            summaries = [
                "AI and machine learning overview covering algorithms, deep learning, and applications.",
                "Vector databases for high-dimensional data storage and semantic search capabilities.",
                "RAG systems combining language models with external knowledge retrieval.",
            ]

        print("Generated summaries:")
        for i, summary in enumerate(summaries):
            print(f"{i + 1}. {summary}")

        # Set up retriever
        vectorstore = Chroma(
            collection_name="summaries", embedding_function=self.embeddings
        )
        store = InMemoryByteStore()
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key="doc_id",
        )

        # Create summary documents
        doc_ids = [str(uuid.uuid4()) for _ in self.docs]
        summary_docs = [
            Document(page_content=s, metadata={"doc_id": doc_ids[i]})
            for i, s in enumerate(summaries)
        ]

        if self.embeddings:
            retriever.vectorstore.add_documents(summary_docs)
            retriever.docstore.mset(list(zip(doc_ids, self.docs)))

        return retriever

    def strategy_3_hypothetical_questions(self):
        """Strategy 3: Generate hypothetical questions, embed questions, return docs"""
        print("\nSTRATEGY 3: HYPOTHETICAL QUESTIONS")
        print("-" * 40)

        # Define structure for LLM output
        class HypotheticalQuestions(BaseModel):
            """Generate hypothetical questions."""

            questions: List[str] = Field(..., description="List of questions")

        if self.llm:
            chain = (
                {"doc": lambda x: x.page_content}
                | ChatPromptTemplate.from_template(
                    "Generate a list of exactly 3 hypothetical questions that the below document could be used to answer:\n\n{doc}"
                )
                | self.llm.with_structured_output(HypotheticalQuestions)
                | (lambda x: x.questions)
            )

            hypothetical_questions = chain.batch(self.docs, {"max_concurrency": 5})
        else:
            # Mock questions for demo
            hypothetical_questions = [
                [
                    "What is artificial intelligence?",
                    "How does machine learning work?",
                    "What are the applications of deep learning?",
                ],
                [
                    "What are vector databases?",
                    "How do vector databases enable semantic search?",
                    "Which vector databases are most popular?",
                ],
                [
                    "What is retrieval-augmented generation?",
                    "How does RAG reduce hallucination?",
                    "What are the benefits of RAG systems?",
                ],
            ]

        print("Generated hypothetical questions:")
        for i, questions in enumerate(hypothetical_questions):
            print(f"Document {i + 1}:")
            for j, q in enumerate(questions):
                print(f"  {j + 1}. {q}")

        # Set up retriever
        vectorstore = Chroma(
            collection_name="hypo-questions", embedding_function=self.embeddings
        )
        store = InMemoryByteStore()
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key="doc_id",
        )

        # Create question documents
        doc_ids = [str(uuid.uuid4()) for _ in self.docs]
        question_docs = []
        for i, question_list in enumerate(hypothetical_questions):
            question_docs.extend(
                [
                    Document(page_content=s, metadata={"doc_id": doc_ids[i]})
                    for s in question_list
                ]
            )

        if self.embeddings:
            retriever.vectorstore.add_documents(question_docs)
            retriever.docstore.mset(list(zip(doc_ids, self.docs)))

        return retriever

    def compare_retrieval_methods(self):
        """Compare how different strategies retrieve information"""
        print("\nRETRIEVAL COMPARISON")
        print("=" * 50)

        query = "How does semantic search work?"
        print(f"Query: '{query}'\n")

        strategies = [
            ("Chunks", "Matches against small document pieces"),
            ("Summaries", "Matches against document summaries"),
            ("Questions", "Matches against hypothetical questions"),
        ]

        for strategy, description in strategies:
            print(f"{strategy} Strategy:")
            print(f"  - {description}")
            print("  - Returns: Full parent document")
            print("  - Good for: Different types of queries\n")

    def run_demo(self):
        """Run the complete demonstration"""
        print("LANGCHAIN MULTI-VECTOR RETRIEVER DEMO")
        print("=" * 50)

        print("Sample Documents:")
        for i, doc in enumerate(self.docs, 1):
            print(f"{i}. Topic: {doc.metadata['topic']}")
            print(f"   Content: {doc.page_content.strip()[:100]}...")
            print()

        # Demonstrate all three strategies
        self.strategy_1_smaller_chunks()
        self.strategy_2_summaries()
        self.strategy_3_hypothetical_questions()
        self.compare_retrieval_methods()

        print("IMPLEMENTATION NOTES:")
        print("-" * 30)
        print("1. MultiVectorRetriever handles the complexity of:")
        print("   - Storing multiple vectors per document")
        print("   - Mapping retrieved vectors to parent documents")
        print("   - Returning full documents instead of fragments")
        print()
        print("2. Key components:")
        print("   - VectorStore: Stores and searches embeddings")
        print("   - ByteStore: Stores original documents")
        print("   - ID mapping: Links vectors to parent documents")
        print()
        print("3. Benefits over single-vector approach:")
        print("   - Better granular matching")
        print("   - Multiple representation strategies")
        print("   - Flexible retrieval based on content type")


# Additional utility functions for advanced multi-vector techniques


class AdvancedMultiVectorTechniques:
    """Advanced techniques for multi-vector embeddings"""

    @staticmethod
    def hierarchical_chunking(document: str, levels: List[int] = [1000, 500, 200]):
        """Create hierarchical chunks at different granularities"""
        chunks_by_level = {}

        for level, chunk_size in enumerate(levels):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_size // 10
            )

            docs = [Document(page_content=document)]
            chunks = splitter.split_documents(docs)
            chunks_by_level[f"level_{level}"] = [chunk.page_content for chunk in chunks]

        return chunks_by_level

    @staticmethod
    def create_contextual_embeddings(chunks: List[str], document_context: str):
        """Add document context to chunks for better embeddings"""
        contextual_chunks = []

        for chunk in chunks:
            # Add document context as prefix
            contextual_chunk = (
                f"Document context: {document_context[:100]}...\n\nChunk: {chunk}"
            )
            contextual_chunks.append(contextual_chunk)

        return contextual_chunks

    @staticmethod
    def weighted_vector_combination(vectors: List[List[float]], weights: List[float]):
        """Combine multiple vectors with weights"""
        if len(vectors) != len(weights):
            raise ValueError("Number of vectors must match number of weights")

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Weighted combination
        combined = [0.0] * len(vectors[0])
        for vector, weight in zip(vectors, normalized_weights):
            for i, val in enumerate(vector):
                combined[i] += val * weight

        return combined


# Example usage and comparison function
def main():
    """Main function to demonstrate both approaches"""
    print("COMPLETE VECTOR EMBEDDINGS COMPARISON")
    print("=" * 60)

    # Note: For full functionality, you'd need to set your OpenAI API key
    # demo = LangChainMultiVectorDemo(openai_api_key="your-key-here")
    demo = LangChainMultiVectorDemo()
    demo.run_demo()

    print("\n" + "=" * 60)
    print("ADVANCED TECHNIQUES EXAMPLE")
    print("=" * 60)

    # Demonstrate advanced techniques
    sample_doc = SAMPLE_DOCS[0].page_content
    advanced = AdvancedMultiVectorTechniques()

    # Hierarchical chunking
    hierarchical_chunks = advanced.hierarchical_chunking(sample_doc)
    print("Hierarchical Chunking Results:")
    for level, chunks in hierarchical_chunks.items():
        print(f"{level}: {len(chunks)} chunks")
        if chunks:
            print(f"  Example: {chunks[0][:80]}...")

    # Contextual embeddings
    regular_chunks = hierarchical_chunks["level_2"][:2]
    context = (
        "This document discusses artificial intelligence and machine learning concepts."
    )
    contextual_chunks = advanced.create_contextual_embeddings(regular_chunks, context)

    print("\nContextual Embedding Example:")
    print(f"Original: {regular_chunks[0][:60]}...")
    print(f"With context: {contextual_chunks[0][:100]}...")

    print("\n" + "=" * 60)
    print("WHEN TO USE EACH APPROACH")
    print("=" * 60)
    print("""
Single Vector Embeddings:
✓ Simple documents with uniform content
✓ Fast retrieval requirements
✓ Limited storage/compute resources
✓ Straightforward semantic matching

Multi-Vector Embeddings:
✓ Complex, long documents
✓ Documents with multiple topics/sections
✓ Need for granular retrieval
✓ Diverse query types (questions, summaries, keywords)
✓ When accuracy is more important than speed

Specific Multi-Vector Strategies:
• Chunking: Best for long documents, technical content
• Summaries: Good for overview queries, topic matching
• Hypothetical Questions: Excellent for Q&A systems, FAQ matching
• Hierarchical: When you need multiple levels of detail
    """)


if __name__ == "__main__":
    main()
