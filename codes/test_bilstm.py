import torch
import traceback

from utils import BidirectionalLSTM, encode_text, pad_sequence_to_length
from attention.bahdanau import BahdanauAttention
from attention.luongconcat import LuongConcatAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ── Shared assets ──────────────────────────────────────────────────────────
word2idx        = torch.load("word2idx.pt",        map_location=device)
embedding_matrix = torch.load("embedding_matrix.pt", map_location=device)
idx2word        = {idx: word for word, idx in word2idx.items()}

HIDDEN_SIZE = 64
OUTPUT_SIZE = 2
ATTN_DIM    = 32
DROPOUT     = 0.3
MAX_LEN     = 50

COMBOS = [
    ("BiLSTM + Bahdanau",     BahdanauAttention,    "../models/BiLSTM with Bahdanau.pth"),
    ("BiLSTM + LuongConcat",  LuongConcatAttention, "../models/BiLSTM with LuongConcat.pth"),
]

TEXT = "This movie was absolutely fantastic and I loved every moment of it!"

# ── Encode input ───────────────────────────────────────────────────────────
tokens        = encode_text(TEXT, word2idx)
padded_tokens = pad_sequence_to_length(tokens, MAX_LEN, word2idx["<PAD>"])
input_tensor  = torch.tensor([padded_tokens], dtype=torch.long).to(device)

words = [
    idx2word.get(idx, "<UNK>")
    for idx in padded_tokens
    if idx != word2idx["<PAD>"]
]

print(f"Input text : {TEXT}")
print(f"Tokens     : {words}\n")
print("=" * 60)

# ── Test each combo ────────────────────────────────────────────────────────
for label, AttnClass, weight_path in COMBOS:
    print(f"\n[TEST] {label}")
    print(f"  Weight file: {weight_path}")
    try:
        # Build model
        model = BidirectionalLSTM(
            embedding_matrix=embedding_matrix,
            hidden_size=HIDDEN_SIZE,
            output_size=OUTPUT_SIZE,
            attention_class=AttnClass,
            attn_dim=ATTN_DIM,
            dropout=DROPOUT,
        )

        # Load weights
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("  Weights loaded successfully")

        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)

        if isinstance(output, tuple):
            logits, attn_weights = output
            attn_weights = attn_weights.cpu().numpy()[0][:len(words)].tolist()
        else:
            logits = output
            attn_weights = None

        probs      = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

        print(f"  Forward pass OK")
        print(f"  Sentiment  : {'Positive' if prediction == 1 else 'Negative'}")
        print(f"  Confidence : {confidence:.4f}")
        if attn_weights:
            print(f"  Attn sample: {[round(w, 3) for w in attn_weights[:5]]} ...")

    except Exception:
        print(f"  FAILED:")
        traceback.print_exc()

print("\n" + "=" * 60)
print("Done.")
