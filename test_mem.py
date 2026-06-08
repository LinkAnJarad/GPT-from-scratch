import torch
from training.model import LanguageModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LanguageModel(
    vocab_size=128256, dim=1024, hidden_dim=2816, n_heads=16, n_kv_heads=8,
    n_layers=20, seq_len=4096, local_att_every=-1, window=128, dropout=0.1
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

x = torch.randint(0, 128256, (8, 4096), device=device)
y = torch.randint(0, 128256, (8, 4096), device=device)
loss_fn = torch.nn.CrossEntropyLoss()

scaler = torch.cuda.amp.GradScaler()
torch.cuda.reset_peak_memory_stats()
with torch.autocast("cuda", dtype=torch.float16):
    logits = model(x)
    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
scaler.scale(loss).backward()

print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
