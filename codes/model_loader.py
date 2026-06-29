import os
import onnxruntime as ort

# ---------------------------------------------------------------------------
# ONNX weight-file lookup table.
# Keys mirror the old PyTorch WEIGHT_PATHS so main.py needs minimal changes.
# ---------------------------------------------------------------------------
_ONNX_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "onnx")

WEIGHT_PATHS = {
    ("vanilla_lstm",       "luong_general"): os.path.join(_ONNX_DIR, "vanilla_lstm__luong_general.onnx"),
    ("vanilla_lstm",       "bahdanau"):      os.path.join(_ONNX_DIR, "vanilla_lstm__bahdanau.onnx"),
    ("bidirectional_lstm", "bahdanau"):      os.path.join(_ONNX_DIR, "bidirectional_lstm__bahdanau.onnx"),
    ("bidirectional_lstm", "luong_concat"):  os.path.join(_ONNX_DIR, "bidirectional_lstm__luong_concat.onnx"),
}

# Cache: (model_name, attention_name) → ort.InferenceSession
_session_cache: dict = {}


def get_model(model_name: str, attention_name: str, embedding_matrix=None):
    """
    Return a cached ONNX Runtime InferenceSession for the requested
    (model_name, attention_name) combination.

    `embedding_matrix` is accepted for API compatibility but ignored —
    the embedding weights are already baked into the ONNX graph.

    Raises KeyError if the combination has no exported ONNX file.
    """
    key = (model_name, attention_name)

    if key in _session_cache:
        return _session_cache[key]

    if key not in WEIGHT_PATHS:
        available = ", ".join(f"{m}/{a}" for m, a in WEIGHT_PATHS)
        raise KeyError(
            f"No ONNX model for model='{model_name}', "
            f"attention='{attention_name}'. "
            f"Available combinations: {available}"
        )

    onnx_path = WEIGHT_PATHS[key]
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"ONNX file not found: {onnx_path}. "
            "Run codes/export_onnx.py first."
        )

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    _session_cache[key] = session
    return session