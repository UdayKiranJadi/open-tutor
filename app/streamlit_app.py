import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.generator import GenerationSettings, generate_response
from app.model_loader import load_model_and_tokenizer
from app.prompts import build_prompt


@st.cache_resource
def load_llm():
    """
    Load the Hugging Face model only once.
    Streamlit reruns the script often, so caching prevents reloading every time.
    """
    return load_model_and_tokenizer()


def get_mode_key(mode_label: str) -> str:
    """
    Convert user-friendly mode label into internal mode key.
    """
    mode_map = {
        "Explain a Concept": "explain",
        "Summarize Text": "summarize",
        "Generate Quiz": "quiz",
        "Rewrite Notes": "rewrite",
        "Ask Anything": "ask",
    }

    return mode_map[mode_label]


def get_placeholder(mode_label: str) -> str:
    """
    Return helpful placeholder text based on selected mode.
    """
    placeholders = {
        "Explain a Concept": "Example: Retrieval Augmented Generation",
        "Summarize Text": "Paste text you want summarized...",
        "Generate Quiz": "Paste study material to generate quiz questions...",
        "Rewrite Notes": "Paste messy notes you want rewritten clearly...",
        "Ask Anything": "Example: What is the difference between RAG and fine-tuning?",
    }

    return placeholders[mode_label]


def get_example_input(mode_label: str) -> str:
    """
    Return example input for each mode.
    """
    examples = {
        "Explain a Concept": "Retrieval Augmented Generation",
        "Summarize Text": (
            "RAG is a technique where an AI system retrieves relevant information "
            "from an external knowledge base before generating an answer. This helps "
            "reduce hallucinations and improves factual accuracy."
        ),
        "Generate Quiz": (
            "Embeddings are numerical representations of text. They help computers "
            "compare meaning. Vector databases store embeddings and allow semantic "
            "similarity search."
        ),
        "Rewrite Notes": (
            "rag means retrieve plus generate. embeddings turn text into numbers. "
            "vector db finds similar chunks. llm uses chunks to answer."
        ),
        "Ask Anything": "What is the difference between RAG and fine-tuning?",
    }

    return examples[mode_label]


def initialize_session_state() -> None:
    """
    Initialize Streamlit session state variables.
    """
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    if "history" not in st.session_state:
        st.session_state.history = []


def add_to_history(mode_label: str, user_input: str, response: str) -> None:
    """
    Store generated response in session history.
    """
    st.session_state.history.insert(
        0,
        {
            "mode": mode_label,
            "input": user_input,
            "response": response,
        }
    )


st.set_page_config(
    page_title="OpenTutor",
    page_icon="🧠",
    layout="wide"
)

initialize_session_state()

st.title("🧠 OpenTutor")
st.caption("A study assistant powered by an open-source Hugging Face LLM.")

with st.sidebar:
    st.header("Model Settings")

    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=50,
        max_value=500,
        value=180,
        step=10
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.6,
        step=0.1
    )

    top_p = st.slider(
        "Top-p",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05
    )

    repetition_penalty = st.slider(
        "Repetition penalty",
        min_value=1.0,
        max_value=1.5,
        value=1.1,
        step=0.05
    )

    st.markdown("---")

    st.header("Tips")
    st.markdown(
        """
        **Lower temperature** gives safer answers.  
        **Higher temperature** gives more creative answers.  
        **More tokens** means longer responses.
        """
    )

    st.markdown("---")

    if st.button("Clear Response History"):
        st.session_state.history = []
        st.success("History cleared.")


with st.spinner("Loading open-source model... This may take a minute the first time."):
    model, tokenizer, device = load_llm()

st.success(f"Model loaded successfully on `{device}`")

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Create with OpenTutor")

    mode_label = st.selectbox(
        "Choose a mode",
        [
            "Explain a Concept",
            "Summarize Text",
            "Generate Quiz",
            "Rewrite Notes",
            "Ask Anything",
        ]
    )

    example_text = get_example_input(mode_label)

    if st.button("Use Example Input"):
        st.session_state.user_input = example_text

    user_input = st.text_area(
        "Enter your text or question",
        placeholder=get_placeholder(mode_label),
        height=240,
        key="user_input"
    )

    generate_button = st.button("Generate Response", type="primary")

with right_col:
    st.subheader("Current Mode")
    st.info(mode_label)

    st.markdown("### Example")
    st.code(example_text)

    st.markdown("### Best Used For")

    if mode_label == "Explain a Concept":
        st.write("Learning new concepts in simple language.")
    elif mode_label == "Summarize Text":
        st.write("Condensing long text into key points.")
    elif mode_label == "Generate Quiz":
        st.write("Creating study questions from notes.")
    elif mode_label == "Rewrite Notes":
        st.write("Cleaning messy notes into organized format.")
    elif mode_label == "Ask Anything":
        st.write("General questions and explanations.")


if generate_button:
    if not user_input.strip():
        st.warning("Please enter some text or a question.")
    else:
        mode = get_mode_key(mode_label)

        prompt = build_prompt(
            mode=mode,
            user_input=user_input
        )

        settings = GenerationSettings(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

        with st.spinner("Generating response..."):
            response = generate_response(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                device=device,
                settings=settings
            )

        add_to_history(
            mode_label=mode_label,
            user_input=user_input,
            response=response
        )

        st.subheader("Response")
        st.markdown(response)

        with st.expander("View Generated Prompt"):
            st.code(prompt)


st.markdown("---")
st.subheader("Response History")

if not st.session_state.history:
    st.info("No responses yet. Generate one to see it here.")
else:
    for i, item in enumerate(st.session_state.history, start=1):
        with st.expander(f"{i}. {item['mode']}"):
            st.markdown("**Input:**")
            st.write(item["input"])

            st.markdown("**Response:**")
            st.write(item["response"])