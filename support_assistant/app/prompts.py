# This module defines a structured prompt template for the Retrieval-Augmented Generation (RAG) model used in Zepto's AI Customer Support Assistant. 
# The prompt is designed to ensure that the model provides accurate and concise answers based solely on the provided context from Zepto's knowledge base.

STRUCTURED_RAG_PROMPT_TEMPLATE = """\
[ROLE]
You are Zepto's AI Customer Support Assistant. Your primary goal is to provide accurate, concise, and helpful answers regarding Zepto's delivery, returns, membership, tracking, cancellation, damaged item, gift card, and customer support policies.
[CONTEXT]
The following policy documents retrieved from Zepto's knowledge base are provided to ground your answer:
{context}
[TASK]
Answer the user query accurately using ONLY the information provided in the [CONTEXT] block above.
[LENGTH]
Keep the answer under 100 words
[NEGATIVE CONSTRAINTS]
1. Do not answer using information not present in the provided context.
2. Do not speculate, extrapolate, or assume policies that are not explicitly stated.
3. If the answer cannot be fully determined from the context, state clearly: "I cannot answer this question based on Zepto's available policies."
[FORMAT]
Respond strictly with a valid JSON object adhering to the following schema:
{{
  "answer": "<your clear, direct answer string>",
  "sources": ["<list of source doc_ids used, e.g., 'doc_01'>"],
  "confidence": <float between 0.0 and 1.0 based on context relevance>
}}

[FEW-SHOT EXAMPLES]
Example 1:
Context:
[doc_01]: Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.
User Query: What is the delivery fee for a 100 rupee order?
Output:
{{
  "answer": "For orders under INR 149, Zepto charges a flat delivery fee of INR 25.",
  "sources": ["doc_01"],
  "confidence": 1.0
}}

Example 2:
Context:
[doc_08]: Zepto customer support is available via in-app chat 24 hours a day, 7 days a week. Phone support is not offered.
User Query: Can I call Zepto support on the phone?
Output:
{{
  "answer": "No, Zepto does not offer phone support. Support is available 24/7 via in-app chat.",
  "sources": ["doc_08"],
  "confidence": 0.95
}}
User Query: {query}
"""

def build_prompt(chunks: list[dict], query: str) -> str:
    """
    Constructs a structured prompt for the RAG model based on the provided context chunks and user query.

    Args:
        chunks (list[dict]): Retrieved chunks, each with an 'id' (e.g. "doc_07") and 'text'.
        query (str): The user's query that needs to be answered.

    Returns:
        str: A formatted prompt string ready to be sent to the LLM.
    """
    # Format the context chunks with their real document IDs, e.g. [doc_07]: ...
    formatted_chunks = "\n".join(f"[{chunk['id']}]: {chunk['text']}" for chunk in chunks)

    # Fill in the prompt template with the formatted context and user query
    prompt = STRUCTURED_RAG_PROMPT_TEMPLATE.format(context=formatted_chunks, query=query)

    return prompt

if __name__ == "__main__":
    # Test the build_prompt function with two made-up chunks and a sample question
    test_chunks = [
        {"id": "doc_01", "text": "Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee."},
        {"id": "doc_08", "text": "Zepto customer support is available via in-app chat 24 hours a day, 7 days a week. Phone support is not offered."},
    ]
    test_query = "What is the delivery fee for a 100 rupee order?"
    
    # Build the prompt
    test_prompt = build_prompt(test_chunks, test_query)
    
    # Print the generated prompt
    print(test_prompt)