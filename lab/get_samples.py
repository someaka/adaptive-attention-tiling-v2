from datasets import load_dataset, Dataset
import random
import json
import os
from typing import List, Dict, Any, cast

def get_samples(num_samples: int = 10, max_length: int = 128) -> List[str]:
    # Load Wikitext-2 dataset
    dataset = cast(Dataset, load_dataset("wikitext", "wikitext-2-raw-v1", split="train"))
    
    # Get the training split
    train_data = cast(List[Dict[str, str]], dataset)
    
    # Filter out empty strings and very short sequences
    valid_texts = [item['text'] for item in train_data 
                  if isinstance(item.get('text', ''), str) and 
                  len(item.get('text', '').strip()) > max_length]
    
    # Randomly sample sequences
    samples = random.sample(valid_texts, num_samples)
    
    # Trim to max_length if needed
    samples = [text[:max_length] for text in samples]
    
    return samples

if __name__ == "__main__":
    samples = get_samples()
    
    # Save samples to file
    samples_file = "samples.json"
    with open(samples_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"=== Saved {len(samples)} samples to {samples_file} ===\n")
    for i, sample in enumerate(samples, 1):
        print(f"Sample {i}:")
        print(f"{sample}\n") 