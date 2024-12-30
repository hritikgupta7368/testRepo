import random fugvjhv,jhv

def generate_random_list(size):
    """Generate a list of random numbers"""
    return [random.randint(1, 100) for _ in range(size)]

def find_average(numbers):cdcd
    """Calculate the average of a list of numbers"""
    return sum(numbers) / len(numbers) if numbers else 0

if __name__ == "__main__":
    # Generate a random list of 5 numbers
    random_numbers = generate_random_list(5)
    print("Random numbers:", random_numbers)
    
    # Calculate and print the average
    avg = find_average(random_numbers)
    print(f"Average: {avg:.2f}")
    # Find maximum and minimum values
    max_value = max(random_numbers)
    min_value = min(random_numbers)
    print(f"Maximum value: {max_value}")
    print(f"Minimum value: {min_value}")
    
    # Calculate variance
    variance = sum((x - avg) ** 2 for x in random_numbers) / len(random_numbers)
    print(f"Variance: {variance:.2f}")
    
    # Sort the list
    sorted_numbers = sorted(random_numbers)
    print("Sorted numbers:", sorted_numbers)
    # Add transformer-based text generation model
    class SimpleTransformer:
        def __init__(self, vocab_size=1000, d_model=512, nhead=8, num_layers=6):
            self.embedding = torch.nn.Embedding(vocab_size, d_model)
            self.pos_encoding = self._create_positional_encoding(d_model)
            self.transformer = torch.nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers
            )
            self.fc_out = torch.nn.Linear(d_model, vocab_size)
            
        def _create_positional_encoding(self, d_model, max_seq_length=5000):
            pos = torch.arange(max_seq_length).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_seq_length, d_model)
            pe[:, 0::2] = torch.sin(pos * div_term)
            pe[:, 1::2] = torch.cos(pos * div_term)
            return pe
        
        def forward(self, src, tgt):
            src = self.embedding(src) + self.pos_encoding[:src.size(0)]
            tgt = self.embedding(tgt) + self.pos_encoding[:tgt.size(0)]
            out = self.transformer(src, tgt)
            return self.fc_out(out)

    # Initialize model and generate text from numbers
    model = SimpleTransformer()
    text_representation = model(torch.tensor(random_numbers).unsqueeze(0), 
                              torch.tensor(sorted_numbers).unsqueeze(0))
    print("Transformer output shape:", text_representation.shape)