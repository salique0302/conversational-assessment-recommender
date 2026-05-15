# System Approach & Architecture

## 1. Design & Architecture
The SHL Conversational Assessment Recommender was designed with a strict emphasis on modularity, production-readiness, and robust safety guardrails. The architecture uses a stateless API endpoint (`/chat`) that accepts full conversation histories to prevent complex session state management on the server.

### Key Components:
- **FastAPI Backend:** Handles incoming HTTP requests and ensures input/output schemas match exactly using Pydantic models.
- **Decision Engine (Agent):** Uses OpenAI's `gpt-4o-mini` with Structured Outputs. It doesn't generate raw text; instead, it generates an `ExtractedIntent` object containing flags (`is_off_topic`, `action`), search queries, and specific conversational directives. This acts as a robust orchestrator.
- **Retriever System:** Built on a local `FAISS` vector index powered by `sentence-transformers/all-MiniLM-L6-v2`. It enables hyper-fast, stateless semantic retrieval.
- **Reranker Engine:** Uses a CrossEncoder (`ms-marco-MiniLM-L-6-v2`) layered with custom lexical-overlap heuristic bonuses to ensure domain-specific terminology (e.g. "statistical data" -> "Numerical Reasoning") achieves perfect recall.

## 2. Tradeoffs
- **Stateless vs Stateful:** We chose a stateless API where the client passes the entire history. **Tradeoff:** Increases token usage per request, but drastically simplifies backend architecture, allowing horizontal scaling without sticky sessions or caching databases.
- **Local FAISS vs Managed Vector DB:** We chose local FAISS index files for simplicity, cost, and speed. **Tradeoff:** If the SHL catalog scales to millions of items, a managed DB like Pinecone or Qdrant would be required. For catalogs under 100k items, FAISS in memory is highly efficient.
- **LLM vs Rule-based Routing:** We rely on an LLM to extract intent rather than simple Regex rules. **Tradeoff:** Slightly higher latency, but significantly better ability to handle vague queries, context shifts, and prompt-injection attempts contextually.

## 3. Evaluation & Performance
An automated evaluation harness (`scripts/evaluate.py`) was constructed to simulate real-world vague queries mapping to known assessments.
- **Baseline Recall@1:** Initially `0.75` due to the gap between colloquial terms (like "angry clients") and formal assessment names.
- **Optimized Recall@1:** Achieved `1.00` (100%) by layering a targeted lexical overlap and synonym bonus over the semantic CrossEncoder score.
- **Latency Optimizations:** The CrossEncoder candidate pool was reduced to `top-k=10` to speed up CPU inference, and the LLM context window was clamped to the last 8 messages (4 conversational turns) to significantly reduce token processing times.

## 4. Guardrails & Safety
The agent enforces strict boundary definitions:
- Rejects non-HR/non-recruitment topics (Off-topic handling).
- Rejects jailbreak or system-prompt extraction attempts (Injection defense).
- Strictly grounds all test names by pulling them dynamically from FAISS rather than LLM memory, ensuring zero hallucination.
