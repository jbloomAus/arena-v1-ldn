# %% 
import torch as t
import torch.nn.functional as F
import transformers

gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")


# %%
def apply_sampling_methods(
    input_ids: t.Tensor, logits: t.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0
) -> int:
    '''
    Return the next token, sampled from the model's probability distribution with modifiers.
x
    input_ids: shape (seq,)
    '''
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0, "Temperature should be non-negative"
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

    if temperature == 0:
        return greedy_search(logits)
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if freq_penalty != 0.0:
        logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)

def sample_tokens(
    model,
    tokenizer,
    initial_text: str,
    max_tokens_generated: int = 30,
    **kwargs
) -> str:
    '''
    Sample tokens until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    model.eval()
    input_ids: list = tokenizer.encode(initial_text)
    generated = []
    device = next(model.parameters()).device
    for _ in range(max_tokens_generated):
        new_input_ids = t.tensor(input_ids + generated, dtype=t.int64, device=device)
        new_input_ids_truncated = new_input_ids[-min(tokenizer.model_max_length, new_input_ids.shape[0]):].unsqueeze(0)
        output = model(new_input_ids_truncated)
        all_logits = output if isinstance(output, t.Tensor) else output.logits
        logits = all_logits[0, -1]
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        generated.append(new_token)
        if new_token == getattr(tokenizer, "eos_token_id", None):
            break
    return tokenizer.decode(input_ids + generated)



# %% 
def greedy_search(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, )

    Return: the most likely token (as an integer)
    '''
    return logits.argmax().item()

prompt = "Jingle bells, jingle bells, jingle all the way"
print("Greedy decoding with prompt: ", prompt)
output = sample_tokens(gpt, tokenizer, prompt, max_tokens_generated=8, temperature=0.0)
print(f"Your model said: {output}")
expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
assert output == expected

print("Greedy decoding a second time (should be deterministic): ")
output = sample_tokens(gpt, tokenizer, prompt, max_tokens_generated=8, temperature=0.0)
print(f"Your model said: {output}")
expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
assert output == expected

print("Tests passed!")

# %% 
def sample_basic(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    return t.multinomial(t.softmax(logits, dim = 0), num_samples=1).item()


N = 20000
probs = t.linspace(0, 0.4, 5)
unnormalized_logits = probs.log() + 1.2345
samples = t.tensor([sample_basic(unnormalized_logits) for _ in range(N)])
counts = t.bincount(samples, minlength=len(probs)) / N
print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
t.testing.assert_close(counts, probs, atol=0.01, rtol=0)
print("Tests passed!")

# %%
def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    '''
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    assert temperature > 0
    logits = logits / temperature
    return logits

logits = t.tensor([1, 2]).log()
cold_logits = apply_temperature(logits, 0.001)
print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
t.testing.assert_close(cold_logits, 1000.0 * logits)
hot_logits = apply_temperature(logits, 1000.0)
print("A high temperature flattens the distribution: ", hot_logits)
t.testing.assert_close(hot_logits, 0.001 * logits)
print("Tests passed!")

# %%
import torch.nn
def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    '''
    input_ids: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    frequencies = t.bincount(input_ids, minlength=logits.shape[0])
    logits = logits- frequencies*freq_penalty
    return logits


bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
input_ids = tokenizer.encode(bieber_prompt, return_tensors="pt").squeeze()
logits = t.ones(tokenizer.vocab_size)
penalized_logits = apply_freq_penalty(input_ids, logits, 2.0)
assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space"
assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space"
print("Tests passed!")
# %%
N_RUNS = 1
your_prompt = "Jingle bells, jingle bells, jingle all the way"
cases = [
    ("High freq penalty", dict(freq_penalty=100.0)),
    ("Negative freq penalty", dict(freq_penalty=-1.0)),
    ("Too hot!", dict(temperature=2.0)),
    ("Pleasantly cool", dict(temperature=0.7)),
    ("Pleasantly warm", dict(temperature=0.9)),
    ("Too cold!", dict(temperature=0.01)),
]
for (name, kwargs) in cases:
    for i in range(N_RUNS):
        output = sample_tokens(gpt, tokenizer, your_prompt, max_tokens_generated=24, **kwargs)
        print(f"Sample {i} with: {name} ({kwargs}):")
        print(f"Your model said: {repr(output)}\n")
# %%
def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    '''

    non_zero_indices = t.topk(logits, top_k).indices
    distribution = t.distributions.categorical.Categorical(logits = logits[non_zero_indices])
    return non_zero_indices[distribution.sample()] #t.multinomial(probs, num_samples=1).item()

N = 20000
k = 3
probs = t.linspace(0, 0.4, 5)
unnormalized_logits = probs.log() + 1.2345
counts = t.bincount(samples, minlength=len(probs)) / N
expected = probs.clone()
expected[:-k] = 0
expected /= expected.sum()
print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
print(expected)
t.testing.assert_close(counts, expected, atol=0.01, rtol=0)
print("Tests passed!")


# %%
your_prompt = "27 years ago, a which done put a spell on me, now everytime I'm speaking, I'm"
output = sample_tokens(gpt, tokenizer, your_prompt, temperature=0.7, top_k=40, max_tokens_generated=64)
print(f"Your model said: {repr(output)}")



# %%
def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    probs = t.softmax(logits, dim = 0)
    values, indices = t.sort(probs, descending=True)
    num_above_p = t.sum(t.cumsum(values,0) < top_p)
    if num_above_p < min_tokens_to_keep:
        indices = t.topk(logits, min_tokens_to_keep).indices
    else: 
        indices = indices[:num_above_p+1]

    distribution = t.distributions.categorical.Categorical(logits = logits[indices])
    return indices[distribution.sample()] #t.multinomial(probs, num_samples=1).item()

N = 4000
unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
samples = t.tensor([sample_top_p(unnormalized_logits, 0.5) for _ in range(N)])
counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
print("top_p of 0.5 or lower should only return token 2: ", counts)
assert counts[0] == 0 and counts[1] == 0

N = 4000
unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
samples = t.tensor([sample_top_p(unnormalized_logits, 0.50001) for _ in range(N)])
counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
print("top_p in (0.5, 0.8] should return tokens 1 and 2: ", counts)
assert counts[0] == 0

N = 4000
top_p = 0.71
probs = t.linspace(0, 0.4, 5)
unnormalized_logits = probs.log() + 1.2345
samples = t.tensor([sample_top_p(unnormalized_logits, top_p) for _ in range(N)])
counts = t.bincount(samples, minlength=len(probs)) / N
expected = probs.clone()
expected[0:2] = 0
expected /= expected.sum()
print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
t.testing.assert_close(counts, expected, atol=0.01, rtol=0.0), 

print("All tests passed!")
# %%
your_prompt = "There is a letter lying on the living-room table, and an unstamped envelope of yellowish parchment, addressed to Mr. H. Potter in emerald-green ink."
output = sample_tokens(gpt, tokenizer, your_prompt, temperature=0.7, top_p=0.95, max_tokens_generated=64)
print(f"Your model said: {repr(output)}")


# %%
def apply_sampling_methods(
    input_ids: t.Tensor, logits: t.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0
) -> int:
    '''
    Return the next token, sampled from the model's probability distribution with modifiers.
x
    input_ids: shape (seq,)
    '''
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0, "Temperature should be non-negative"
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

    if temperature == 0:
        return greedy_search(logits)
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if freq_penalty != 0.0:
        logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)

def sample_tokens(
    model,
    tokenizer,
    initial_text: str,
    max_tokens_generated: int = 30,
    **kwargs
) -> str:
    '''
    Sample tokens until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    model.eval()
    input_ids: list = tokenizer.encode(initial_text)
    generated = []
    device = next(model.parameters()).device
    for _ in range(max_tokens_generated):
        new_input_ids = t.tensor(input_ids + generated, dtype=t.int64, device=device)
        new_input_ids_truncated = new_input_ids[-min(tokenizer.model_max_length, new_input_ids.shape[0]):].unsqueeze(0)
        output = model(new_input_ids_truncated)
        all_logits = output if isinstance(output, t.Tensor) else output.logits
        logits = all_logits[0, -1]
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        generated.append(new_token)
        if new_token == getattr(tokenizer, "eos_token_id", None):
            break
    return tokenizer.decode(input_ids + generated)

# %% sample from model
from src.transformers import TransformerConfig, DecoderOnlyTransformer
from shakespeare_utils import WordDataset,WordsTokenizer

config = {
'batch_size': 16,
'hidden_size': 512,
'lr': 0.00125,
'seq_len': 20,
'num_layers': 6,
'num_heads': 8,
'vocab_size': 34543,
'num_epochs': 5,
'device': 'cpu',
'dropout': 0.1,
'layer_norm_epsilon': 1e-5,
'train_set_size': 2 * 10**4,
'test_set_size': 1000,
'num_workers': 10,
}

shakespeare_text = open('shakespeare.txt', 'r').read()
dataset = WordDataset(shakespeare_text, block_size=20, overwrite_length=2 * 10**4)
tokenizer = WordsTokenizer(dataset)

transformer_config = TransformerConfig(
    num_layers=config["num_layers"],
    num_heads= config["num_heads"],
    vocab_size=config["vocab_size"],
    hidden_size=config["hidden_size"],
    max_seq_len=config["seq_len"],
    dropout=config["dropout"],
    layer_norm_epsilon=config["layer_norm_epsilon"])

print(transformer_config.hidden_size)

model = DecoderOnlyTransformer(config=transformer_config)

model.load_state_dict(t.load("../best_shake_spear_model.pt"))

initial_text = "turn down for what"
text_output = sample_tokens(
    model, 
    tokenizer, 
    initial_text, 
    max_tokens_generated=100, 
    temperature=1.0, 
    top_p = 0.3,
    freq_penalty=2
)

print(text_output)

# %%
tokenizer.encode(text_output)
# %%
