import torch
from model import SimpleTransformer, PreTrainedTransformer
import tiktoken
from transformers import GPT2Tokenizer
from typing import List, Tuple
import argparse

def setup_simple_transformer(checkpoint_path: str = 'lab/model_weights.pt') -> Tuple[SimpleTransformer, tiktoken.Encoding]:
    """Setup our custom transformer model and tokenizer."""
    model = SimpleTransformer(vocab_size=10000)
    checkpoint = torch.load(checkpoint_path)
    if not hasattr(model.quantum_attention, '_return_metrics_buffer'):
        model.quantum_attention.register_buffer('_return_metrics_buffer', torch.tensor(True))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    tokenizer = tiktoken.get_encoding('cl100k_base')
    return model, tokenizer

def setup_distilgpt2() -> Tuple[PreTrainedTransformer, GPT2Tokenizer]:
    """Setup DistilGPT2 model and tokenizer."""
    model = PreTrainedTransformer('distilgpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_simple_transformer(
    model: SimpleTransformer,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 5
) -> str:
    """Generate text using our custom transformer model."""
    with torch.no_grad():
        tokens = torch.tensor([t % 10000 for t in tokenizer.encode(prompt)], dtype=torch.long)
        generated_tokens = tokens.tolist()
        
        for _ in range(max_new_tokens):
            # Get predictions for next token
            predictions = model.predict_next(torch.tensor(generated_tokens))
            probs = torch.tensor([p for _, p in predictions])
            
            # Apply temperature and sample from top-k
            probs = torch.softmax(probs / temperature, dim=0)
            next_token_idx = torch.multinomial(probs[:top_k], 1)[0]
            next_token = predictions[next_token_idx][0]
            
            # Add token to sequence
            generated_tokens.append(next_token)
            
            # Stop if we generate a special token
            if next_token in [0, 1]:  # Common special token IDs
                break
        
        return tokenizer.decode([t % 10000 for t in generated_tokens])

def generate_distilgpt2(
    model: PreTrainedTransformer,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 5,
    no_repeat_ngram_size: int = 2
) -> str:
    """Generate text using DistilGPT2."""
    with torch.no_grad():
        # Convert input to tensor format that HuggingFace expects
        inputs = tokenizer(prompt, return_tensors='pt')
        
        # Generate output
        output_ids = model.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pad_token_id=tokenizer.pad_token_id,
            max_length=len(tokenizer.encode(prompt)) + max_new_tokens,
            num_return_sequences=1,
            temperature=temperature,
            top_k=top_k,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=True
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def format_output(model_name: str, text: str) -> str:
    """Format model output for display."""
    separator = "-" * 50
    return f"\n{separator}\n{model_name}:\n{text}\n{separator}"

def main():
    parser = argparse.ArgumentParser(description='Generate text using transformer models')
    parser.add_argument('--prompt', type=str, default="The quantum state",
                      help='Text prompt to start generation')
    parser.add_argument('--max-tokens', type=int, default=50,
                      help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Sampling temperature (higher = more random)')
    parser.add_argument('--top-k', type=int, default=5,
                      help='Number of top tokens to sample from')
    parser.add_argument('--checkpoint', type=str, default='lab/model_weights.pt',
                      help='Path to model checkpoint')
    args = parser.parse_args()

    # Setup models
    simple_model, simple_tokenizer = setup_simple_transformer(args.checkpoint)
    distil_model, distil_tokenizer = setup_distilgpt2()

    # Generate from both models
    simple_output = generate_simple_transformer(
        simple_model, simple_tokenizer, args.prompt,
        args.max_tokens, args.temperature, args.top_k
    )
    print(format_output("SimpleTransformer", simple_output))

    distil_output = generate_distilgpt2(
        distil_model, distil_tokenizer, args.prompt,
        args.max_tokens, args.temperature, args.top_k
    )
    print(format_output("DistilGPT2", distil_output))

if __name__ == "__main__":
    main()