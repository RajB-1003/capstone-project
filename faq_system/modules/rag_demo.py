from modules.langchain_wrapper import SemanticFAQRetriever
import math
import subprocess

USE_OLLAMA = True
OLLAMA_MODEL = "phi3"   # or "mistral"

def _ollama_generate(prompt: str) -> str:
    # Use 'ollama generate' instead of 'run' to avoid interactive blocking
    try:
        result = subprocess.run(
            ["ollama", "generate", OLLAMA_MODEL],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=60
        )
        
        if result.returncode != 0:
            return f"LLM error: {result.stderr.strip()}"
            
        output = result.stdout.strip()

        if not output:
            return "LLM returned empty response."

        return output

    except Exception as e:
        return f"LLM error: {str(e)}"


def _mock_llm_generate(prompt: str, context: str) -> str:
    """
    Mock LLM generation that simulates a RAG model synthesizing an answer.
    Used to keep the dependency footprint lightweight without requiring
    Ollama or API keys as per 'Option C' in requirements.
    """
    # Simply check if the context is empty.
    if not context.strip():
        return "I could not find a precise answer in the available FAQs."

    # Parse the first answer from the context to form a simulated sensible reply.
    # In a real setup, an LLM would do this dynamically.
    lines = context.strip().split("\n")
    first_answer = ""
    for line in lines:
        if line.startswith("A:"):
            first_answer = line[2:].strip()
            break
            
    if not first_answer:
        return "I could not find a precise answer in the available FAQs."
        
    generated = (
        "Based on our guidelines, here is the answer: \n\n"
        f"{first_answer}\n\n"
        "*(Note: This is a generated summary based strictly on the retrieved context.)*"
    )
    return generated

def generate_rag_answer(query: str, top_k: int = 3, fixtures: tuple = None) -> dict:
    """
    Retrieves top_k FAQ results and passes them to a generation layer
    to create a single natural-language answer.
    """
    # Unpack fixtures: (model, corpus_embeddings, faq_docs, bm25_index, patterns, intent_embeddings)
    model, corpus_embeddings, faq_docs = fixtures[0], fixtures[1], fixtures[2]

    # Use existing wrapper to fetch documents
    retriever = SemanticFAQRetriever(
        model=model,
        corpus_embeddings=corpus_embeddings,
        faq_docs=faq_docs,
        top_k=top_k,
    )
    docs = retriever.get_relevant_documents(query)

    # Build context string
    context = ""
    for doc in docs:
        context += f"Q: {doc.get('question', 'Unknown Question')}\n"
        context += f"A: {doc.get('answer', '')}\n\n"

    prompt = f"""
    You are a university assistant.

    STRICT RULES:
    - Answer ONLY using the provided context
    - Do NOT add outside knowledge
    - If unsure, say: "I could not find a precise answer"

    Be concise and clear.

    Question:
    {query}

    Context:
    {context}

    Final Answer:
    """

    if USE_OLLAMA:
        answer = _ollama_generate(prompt)

        if "LLM error" in answer or "timed out" in answer:
            answer = _mock_llm_generate(prompt, context)
    else:
        answer = _mock_llm_generate(prompt, context)

    return {
        "query": query,
        "generated_answer": answer,
        "source_documents": docs,
        "num_docs_used": len(docs)
    }
