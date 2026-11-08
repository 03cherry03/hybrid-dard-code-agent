"""DARD context selection adapted to Dream's shifted prediction coordinates.

The paper's original/masked-shadow keys are retained at every block position.
Dream predicts token t from row t-1, so two separate predictor banks carry the
appropriate predecessor embeddings without becoming keys themselves. The layout
is original context (P+B), masked shadow context (B), original predictors (B),
and verification predictors (B). This is a shifted-backbone adaptation, not an
author implementation or a claim of paper-equivalent Dream predictions.

Masks specify direct attention edges. They do not promise that a token cannot
influence itself indirectly through other reliable keys across multiple layers.
Unlike the previous strict-context adapter, all-M predictions preserve native
fully bidirectional Dream predictions for the same input canvas.
"""
from __future__ import annotations

M, C, U = 0, 1, 2


def _validate(prefix_len, state, confidence):
    import torch
    p, n = int(prefix_len), int(state.numel())
    if p < 1 or n < 1 or state.ndim != 1 or confidence.shape != state.shape:
        raise ValueError("Require a nonempty prefix and matching 1D state/confidence")
    if state.device != confidence.device:
        raise ValueError("State and confidence must reside on the same device")
    if bool(((state < M) | (state > U)).any()) or not bool(torch.isfinite(confidence).all()):
        raise ValueError("Invalid token state or confidence")
    return p, n


def _original_key_choices(state, confidence):
    """Original-vs-shadow key choice for original and verification queries.

Original M rows condition on all original tokens. U rows condition on original
U tokens and masked replacements elsewhere. C rows additionally use C tokens
with at least their confidence (the paper's >= rule). Verification always uses
the masked replacement at the target position, including for C/U self checks.
"""
    import torch
    n = state.numel()
    reliable = (state == U)[None, :].expand(n, n)
    higher_equal = confidence[None, :] >= confidence[:, None]
    candidate = (state == C)[:, None] & (state == C)[None, :] & higher_equal
    original = reliable | candidate | (state == M)[:, None]
    verification = reliable | candidate
    verification = verification.clone()
    verification[torch.arange(n, device=state.device), torch.arange(n, device=state.device)] = False
    return original, verification


def build_two_stream_visibility(prefix_len: int, state, confidence):
    """Return bool visibility [1,1,P+4B,P+4B], True meaning visible.

Every query sees the trusted prefix and exactly one original-or-shadow key for
each current-block position. Predictor banks are never keys. Prefix context
uses reliable original U keys and masked shadow keys elsewhere, which retains
all current-block positions instead of restricting prefix rows to prefix only.
"""
    import torch
    p, n = _validate(prefix_len, state, confidence)
    original, verification = _original_key_choices(state, confidence)
    choices = torch.cat(((state == U)[None, :].expand(p, n),
                         original, verification, original, verification), dim=0)
    visible = torch.zeros((p + 4*n, p + 4*n), dtype=torch.bool, device=state.device)
    visible[:, :p] = True
    visible[:, p:p+n] = choices
    visible[:, p+n:p+2*n] = ~choices
    return visible[None, None]


def packed_inputs(ids, state, confidence, mask_id: int):
    """Pack real context, masked context and shifted predictor embeddings."""
    import torch
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Dream DARD attention requires one 2D input sequence")
    p, n = _validate(ids.shape[1] - state.numel(), state, confidence)
    if ids.device != state.device:
        raise ValueError("Input IDs and states must reside on the same device")
    shadow = torch.full_like(ids[:, p:], int(mask_id))
    original_queries = ids[:, p-1:p+n-1].clone()
    verifier_queries = original_queries.clone()
    if n > 1:
        _, allowed_original = _original_key_choices(state, confidence)
        target = torch.arange(1, n, device=ids.device)
        verifier_queries[0, 1:] = torch.where(
            allowed_original[target, target-1], verifier_queries[0, 1:], int(mask_id))
    joined = torch.cat((ids, shadow, original_queries, verifier_queries), dim=1)
    query_positions = torch.arange(p-1, p+n-1, device=ids.device)
    positions = torch.cat((torch.arange(p+n, device=ids.device),
                           torch.arange(p, p+n, device=ids.device),
                           query_positions, query_positions))[None]
    return joined, positions


def compact_packed_inputs(ids, state, confidence, mask_id: int, *, rectangular=False):
    """Pack only original predictors that the next M proposal consumes.

    Layout: original/shadow context (P+2B), original M predictors (|M|),
    all verification predictors (B). Predictor banks never become keys, so
    deleting unused C/U original queries preserves every remaining row through
    every transformer layer. Absolute predecessor coordinates are unchanged.
    ``rectangular`` omits the all-false predictor key columns as well; the
    caller must use an attention implementation supporting context-only K/V.
    Returns joined IDs, positions, visibility, and original target indices.
    """
    import torch
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Dream DARD attention requires one 2D input sequence")
    p, n = _validate(ids.shape[1] - state.numel(), state, confidence)
    if ids.device != state.device:
        raise ValueError("Input IDs and states must reside on the same device")
    original, verification = _original_key_choices(state, confidence)
    masked = torch.where(state == M)[0]
    predecessor_ids = ids[:, p-1:p+n-1]
    verifier_ids = predecessor_ids.clone()
    if n > 1:
        targets = torch.arange(1, n, device=ids.device)
        verifier_ids[0, 1:] = torch.where(
            verification[targets, targets-1], verifier_ids[0, 1:], int(mask_id))
    shadow = torch.full_like(ids[:, p:], int(mask_id))
    joined = torch.cat((ids, shadow, predecessor_ids[:, masked], verifier_ids), dim=1)
    predecessor_positions = torch.arange(p-1, p+n-1, device=ids.device)
    positions = torch.cat((torch.arange(p+n, device=ids.device),
                           torch.arange(p, p+n, device=ids.device),
                           predecessor_positions[masked], predecessor_positions))[None]
    choices = torch.cat(((state == U)[None, :].expand(p, n),
                         original, verification, original[masked], verification), dim=0)
    keys = p + 2*n if rectangular else joined.shape[1]
    visible = torch.zeros((joined.shape[1], keys), dtype=torch.bool, device=ids.device)
    visible[:, :p] = True
    visible[:, p:p+n] = choices
    visible[:, p+n:p+2*n] = ~choices
    return joined, positions, visible[None, None], masked
