def build_instruction(prompt, tokenizer):
    """
    Apply the chat template to the user prompt

    Args:
        prompt (str): The user prompt.
        tokenizer: The tokenizer object.

    Returns:
        str: The instruction text.
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    decoder_input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return decoder_input_text