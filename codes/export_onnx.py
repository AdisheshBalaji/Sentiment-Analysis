import os
import sys
import torch
import torch.nn as nn

# -- Make sure local modules are importable ------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from config import OUTPUT_SIZE, DROPOUT, MAX_LEN
from utils import MODEL_REGISTRY, ATTENTION_REGISTRY
from model_loader import WEIGHT_PATHS

# -- Paths ---------------------------------------------------------------------
ONNX_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "onnx")
os.makedirs(ONNX_DIR, exist_ok=True)

device = torch.device("cpu")

# -- Load shared assets --------------------------------------------------------
print("Loading vocab and embedding matrix...")
word2idx = torch.load("word2idx.pt", map_location=device, weights_only=True)
embedding_matrix = torch.load("embedding_matrix.pt", map_location=device, weights_only=True)

print(f"  Vocab size      : {len(word2idx):,}")
print(f"  Embedding shape : {tuple(embedding_matrix.shape)}\n")


# -- Wrapper to guarantee (logits, attn_weights) tuple output ------------------
class _ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        if isinstance(out, tuple):
            logits, attn = out
            if attn.dim() == 3:
                attn = attn.squeeze(1)
            return logits, attn
        
        # No-attention model fallback
        return out, torch.zeros(x.size(0), x.size(1), dtype=torch.float32)


def _infer_dims(state: dict, model_name: str, attention_name: str):
    """Read hidden_size and attn_dim directly from the checkpoint's weight shapes."""
    is_lstm = "lstm" in model_name
    gates = 4 if is_lstm else 1
    
    # Determine the correct state key based on model type
    if "bidirectional" in model_name:
        key = "bilstm.weight_ih_l0" if is_lstm else "birnn.weight_ih_l0"
    else:
        key = "lstm.weight_ih_l0" if is_lstm else "rnn.weight_ih_l0"

    hidden_size = state[key].shape[0] // gates

    # Attn_dim: read from projection weights if present
    if attention_name in ["bahdanau", "luong_concat"]:
        attn_dim = state["attention.v.weight"].shape[1]
    else:
        attn_dim = hidden_size

    return hidden_size, attn_dim


# -- Export loop ---------------------------------------------------------------
for (model_name, attention_name), pth_path in WEIGHT_PATHS.items():
    out_name = f"{model_name}__{attention_name}.onnx"
    out_path = os.path.join(ONNX_DIR, out_name)

    print("-" * 60)
    print(f"Model     : {model_name} | Attention: {attention_name}")
    
    # Load weights and infer dims
    state = torch.load(pth_path, map_location=device, weights_only=True)
    hidden_size, attn_dim = _infer_dims(state, model_name, attention_name)
    print(f"Inferred  : hidden_size={hidden_size}, attn_dim={attn_dim}")

    # Instantiate model
    base_model = MODEL_REGISTRY[model_name](
        embedding_matrix=embedding_matrix,
        hidden_size=hidden_size,
        output_size=OUTPUT_SIZE,
        attention_class=ATTENTION_REGISTRY[attention_name],
        attn_dim=attn_dim,
        dropout=DROPOUT,
    )

    base_model.load_state_dict(state)
    model = _ModelWrapper(base_model).eval()

    # Dummy input for tracing
    dummy_input = torch.zeros(1, MAX_LEN, dtype=torch.long)

    # Export
    torch.onnx.export(
        model,
        (dummy_input,),
        out_path,
        dynamo=False,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits", "attn_weights"],
        dynamic_axes={
            "input":        {0: "batch_size"},
            "logits":       {0: "batch_size"},
            "attn_weights": {0: "batch_size"},
        },
    )
    
    file_mb = os.path.getsize(out_path) / 1e6
    print(f"Exported  : {out_path} ({file_mb:.1f} MB)\n")

print(f"All done! ONNX models are in: {os.path.abspath(ONNX_DIR)}")