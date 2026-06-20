from unsloth import FastLanguageModel
import torch
import argparse

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.
load_in_8bit = False  # Use 8bit quantization to reduce memory usage. Can be False.


def main(prompt):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="azherali/Riazi-8B",  # Choose ANY
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        # token = "YOUR_HF_TOKEN", # HF Token for gated models
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Must add for generation
    )

    from transformers import TextStreamer

    _ = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        temperature=0.6,
        top_p=0.95,
        top_k=20,  # For non thinking
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )


if __name__ == "__main__":
    # i want to get prompt from user and then pass using argparse

    parser = argparse.ArgumentParser(description="Inference script for the model")
    parser.add_argument(
        "--prompt",
        type=str,
        help="The prompt to generate text for",
        required=True,
        default="پانچ بچوں نے 20 چاکلیٹس برابر بانٹیں۔ ہر بچے کو کتنی چاکلیٹس ملیں گی؟",
    )

    args = parser.parse_args()

    main(args.prompt)
