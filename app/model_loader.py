import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import MODEL_NAME


def get_device() -> str:
    """
    Detect the best available device.
    PyTorch requires lowercase device names like: cpu, cuda, mps.
    """
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_model_and_tokenizer():
    """
    Load the Hugging Face tokenizer and open-source LLM.
    """
    device = get_device().lower()

    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device