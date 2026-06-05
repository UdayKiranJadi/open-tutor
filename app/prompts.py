from typing import Literal

PromptMode = Literal[
    "explain",
    "summarize",
    "quiz",
    "rewrite",
    "ask"
]

def build_prompt(mode: PromptMode, user_input: str) -> str:
    """
    Build a prompt based on the selected mode.
    """
    if not user_input.strip():
        raise ValueError("User input cannot be empty.")

    if mode == "explain":
        return build_explain_prompt(user_input)

    if mode == "summarize":
        return build_summarize_prompt(user_input)

    if mode == "quiz":
        return build_quiz_prompt(user_input)

    if mode == "rewrite":
        return build_rewrite_prompt(user_input)

    if mode == "ask":
        return build_ask_prompt(user_input)

    raise ValueError(f"Unsupported prompt mode: {mode}")

def build_explain_prompt(concept: str) -> str:
    """
    Build a prompt for explaining a concept.
    """
    return f"""
You are an AI tutor.

Explain the following concept in simple beginner-friendly language.

Requirements:
- Start with a short definition.
- Use an analogy.
- Give one practical example.
- Keep the explanation clear and concise.

Concept:
{concept}
"""

def build_summarize_prompt(text: str) -> str:
    """
    Build a prompt for summarizing text.
    """
    return f"""
You are an expert summarizer.

Summarize the following text.

Requirements:
- Give a short summary.
- Then list the key points.
- Do not add information that is not in the text.
- Keep it easy to understand.

Text:
{text}
"""

def build_quiz_prompt(text: str) -> str:
    """
    Build a prompt for generating quiz questions.
    """
    return f"""
You are an AI tutor creating study questions.

Create a quiz from the following study material.

Requirements:
- Generate 5 questions.
- Include the answer after each question.
- Use a mix of simple and medium-difficulty questions.
- Keep answers short.

Study Material:
{text}
"""

def build_rewrite_prompt(notes: str) -> str:
    """
    Build a prompt for rewriting messy notes.
    """
    return f"""
You are a helpful study assistant.

Rewrite the following notes to make them clearer and better organized.

Requirements:
- Use headings if useful.
- Use bullet points.
- Keep the original meaning.
- Do not add unrelated information.

Notes:
{notes}
"""

def build_ask_prompt(question: str) -> str:
    """
    Build a general question-answering prompt.
    """
    return f"""
You are a helpful AI assistant.

Answer the following question clearly and directly.

Question:
{question}
"""