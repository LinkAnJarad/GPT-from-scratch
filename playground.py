import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

def main():
    parser = argparse.ArgumentParser(
        description="Playground script to load a Hugging Face model and generate text."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B-Base",
        help="Hugging Face model ID to load (default: gpt2)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Deep learning is a subset of machine learning that",
        help="Prompt to generate text from"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (lower = more deterministic)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k filtering parameter"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) filtering parameter"
    )
    args = parser.parse_args()

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target device: {device.upper()}")
    if device == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    print(f"[*] Loading tokenizer for '{args.model}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Configure padding token if not set (common for gpt2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Loading model '{args.model}' (this might take a moment)...")
    # Load model in half-precision on GPU for faster inference & lower VRAM usage
    model_kwargs = {}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()

    print("\n" + "=" * 50)
    print(f"Prompt: {args.prompt}")
    print("=" * 50)
    print("Generated Output:\n")

    # Encode prompt
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    # Setup streaming helper
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Run generation in a separate thread so streamer can read tokens on the main thread
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream output to terminal
    sys.stdout.write(args.prompt)
    sys.stdout.flush()
    for new_text in streamer:
        sys.stdout.write(new_text)
        sys.stdout.flush()
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
