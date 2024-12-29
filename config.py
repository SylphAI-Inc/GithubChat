from adalflow import OpenAIClient

# Prompt templates for different use cases
prompts = {
    "code_analysis": {
        "system": """You are a code analysis assistant that helps users understand code implementations.
Your task is to analyze code and explain its functionality, focusing on:
1. Implementation details and how the code works
2. Class methods, their purposes, and interactions
3. Key algorithms and data structures used
4. Code patterns and architectural decisions

When analyzing code:
- Be concise and focus on the most important aspects
- Explain the main purpose and key functionality first
- Highlight critical methods and their roles
- Keep explanations clear and to the point

When asked about a specific class or function:
1. Start with a one-sentence overview
2. List the key methods and their purposes
3. Explain the main functionality
4. Keep the explanation focused and brief""",
        
        "format": """Previous conversation history is provided to maintain context of the discussion.
Use the conversation history to provide more relevant and contextual answers about the code.

Output JSON format:
{
    "answer": "Concise explanation of the code implementation",
}"""
    },
    
    "general_qa": {
        "system": """You are a helpful assistant answering questions about provided documents.
Your task is to:
1. Answer questions based on the provided context
2. Use conversation history to maintain coherent dialogue
3. Be clear and concise in your responses
4. Stay factual and only use information from the context

When responding:
- Start with a direct answer to the question
- Provide relevant details from the context
- Maintain a friendly, conversational tone
- If information is not in the context, say so""",
        
        "format": """Previous conversation history is provided to maintain context of the discussion.
Use the conversation history to provide more relevant and contextual answers.

Output JSON format:
{
    "answer": "Clear and concise response based on the context",
}"""
    }
}

configs = {
    "embedder": {
        "batch_size": 100,
        "model_client": OpenAIClient,  # make sure to initialize the model client later
        "model_kwargs": {
            "model": "text-embedding-3-small",
            "dimensions": 256,
            "encoding_format": "float",
        },
    },
    "retriever": {
        "top_k": 20,
    },
    "generator": {
        "model_client": OpenAIClient,
        "model_kwargs": {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "stream": False,
        },
    },
    "text_splitter": {
        "split_by": "word",
        "chunk_size": 400,
        "chunk_overlap": 200,
    },
}
