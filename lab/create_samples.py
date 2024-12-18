import json
import torch
import tiktoken
from datasets import load_dataset, Dataset
import numpy as np
from typing import List, Dict, Any, cast

def create_tokenized_samples(num_samples: int = 100, max_length: int = 100) -> None:
    print("Loading dataset...")
    dataset = cast(Dataset, load_dataset("wikitext", "wikitext-2-raw-v1", split="train"))
    
    print("Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Using GPT-4's encoding
    
    print(f"Creating {num_samples} samples...")
    samples: List[Dict[str, List[int]]] = []
    dataset_size = len(dataset)  # Now safe with explicit Dataset type
    
    for i in range(num_samples):
        # Get a random text sample using numpy instead of torch
        idx = np.random.randint(0, dataset_size)
        text = dataset[idx]['text']
        
        # Skip empty texts
        if not isinstance(text, str) or not text.strip():
            continue
            
        # Tokenize
        tokens = tokenizer.encode(text)
        
        # Truncate if needed
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Create attention mask (1 for all tokens)
        attention_mask = [1] * len(tokens)
        
        # Create sample
        sample = {
            'input_ids': tokens,
            'attention_mask': attention_mask
        }
        samples.append(sample)
        
        if (i + 1) % 10 == 0:
            print(f"Created {i + 1} samples")
            print(f"Sample text: {text[:100]}...")
            print(f"Sample tokens: {tokens[:20]}...")
    
    print("Saving samples...")
    with open('lab/tokenized_samples.json', 'w') as f:
        json.dump(samples, f)
    
    # Also save vocab size for the model
    vocab_info = {
        'vocab_size': tokenizer.n_vocab,
        'encoding_name': tokenizer.name
    }
    with open('lab/vocab_info.json', 'w') as f:
        json.dump(vocab_info, f)
    
    print(f"Done! Vocabulary size: {tokenizer.n_vocab}")

if __name__ == "__main__":
    create_tokenized_samples() 