from app.generator import GenerationSettings, generate_response
from app.model_loader import load_model_and_tokenizer
from app.prompts import build_prompt


def main():
    model, tokenizer, device = load_model_and_tokenizer()

    print("\nModel loaded successfully!")
    print("=" * 60)

    mode = "explain"
    user_input = "Retrieval Augmented Generation"

    prompt = build_prompt(mode, user_input)

    settings = GenerationSettings(
        max_new_tokens=180,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.1
    )

    response = generate_response(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        settings=settings
    )

    print("\nMode:")
    print(mode)

    print("\nUser Input:")
    print(user_input)

    print("\nAssistant:")
    print(response)


if __name__ == "__main__":
    main()