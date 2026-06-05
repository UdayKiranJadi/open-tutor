from dataclasses import dataclass

import torch

from app.config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)


@dataclass
class GenerationSettings:
    """
    Configuration for text generation.
    """
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY


def validate_generation_settings(settings: GenerationSettings) -> None:
    """
    Validate generation settings before running the model.
    """
    if settings.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than 0.")

    if not 0 < settings.temperature <= 2:
        raise ValueError("temperature must be between 0 and 2.")

    if not 0 < settings.top_p <= 1:
        raise ValueError("top_p must be between 0 and 1.")

    if settings.repetition_penalty < 1:
        raise ValueError("repetition_penalty must be at least 1.")


def build_chat_messages(prompt: str) -> list[dict[str, str]]:
    """
    Build chat-style messages for instruction-tuned models.
    """
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    return [
        {
            "role": "system",
            "content": "You are OpenTutor, a helpful AI tutor. Explain clearly, avoid unnecessary detail, and keep answers useful."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]


def clean_response(response: str) -> str:
    """
    Clean model output before showing it to the user.
    """
    response = response.strip()

    unwanted_prefixes = [
        "Assistant:",
        "assistant:",
        "AI:",
        "OpenTutor:"
    ]

    for prefix in unwanted_prefixes:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()

    return response


def generate_response(
    prompt: str,
    model,
    tokenizer,
    device: str,
    settings: GenerationSettings | None = None
) -> str:
    """
    Generate a clean response from the open-source LLM.
    """
    if settings is None:
        settings = GenerationSettings()

    validate_generation_settings(settings)

    messages = build_chat_messages(prompt)

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    ).to(device)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                do_sample=True,
                temperature=settings.temperature,
                top_p=settings.top_p,
                repetition_penalty=settings.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        return clean_response(response)

    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            return "The model ran out of memory. Try reducing max_new_tokens or using a smaller model."

        return f"Generation failed: {error}"