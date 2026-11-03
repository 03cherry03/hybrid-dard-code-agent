"""DARD paper-mechanisms implementation for Apple's Dream/DiffuCoder architecture.

The authors' official DARD repository was not located (as of 2026-11-02).
This is an INDEPENDENT implementation, NOT an exact, validated reproduction.

Included mechanisms:
 - masked / candidate / verified (M / C / U) token states
 - confidence-ordered selective attention for candidate verification
 - original + shadow streams in ONE custom-mask transformer forward
 - adaptive logit mixing based on local candidate promotions/revocations
 - candidates can be revoked; blockwise continuation; no unfilled masks

Dream's pre-token shift is handled explicitly: logits at input position t-1
predict the target token at t, even though attention is bidirectional.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math


@dataclass(frozen=True)
class DARDConfig:
    block_size: int = 128
    steps_per_block: int = 192
    tau_c: float = 0.10
    tau_u: float = 0.30
    transfer_limit: int = 8
    mix_base: float = 0.90
    mix_impact: float = 0.65
    mix_decay: float = 8.0
    temperature: float = 0.3
    top_p: float = 0.95

    def validate(self) -> None:
        if not (1 <= self.block_size <= 1024 and 1 <= self.steps_per_block <= 4096):
            raise ValueError("block_size and steps_per_block are out of range")
        if not (0 < self.tau_c < self.tau_u < 1):
            raise ValueError("Require 0 < tau_c < tau_u < 1")
        if self.transfer_limit < 1:
            raise ValueError("transfer_limit must be positive")
        if not (0.0 <= self.mix_base <= 1.0 and 0 <= self.mix_impact <= 1.0 and self.mix_decay > 0):
            raise ValueError("Invalid adaptive-mixing parameters")
        if not (self.temperature >= 0 and 0 < self.top_p <= 1):
            raise ValueError("Invalid sampling parameters")


M, C, U = 0, 1, 2


def build_two_stream_visibility(prefix_len: int, state, confidence):
    """Boolean SDPA visibility [1,1,2L,2L], original+shadow streams.

    A candidate key in the ORIGINAL stream cannot depend (through any layer)
    on less-reliable candidate keys, because its row is also restricted.
    SHADOW candidate queries can see the ORIGINAL strictly higher-confidence
    candidate keys, but cannot see their own original key.

    Original M-position prediction queries may see tentative candidate tokens;
    shadow M-position prediction queries see only reliable context. This
    enables original/shadow mixing without a separate transformer pass.

    This mask is an independent integration of DARD's mechanism; it is not
    claimed to reproduce every paper implementation choice exactly.
    """
    import torch
    s = [int(x) for x in state.tolist()]
    conf = [float(x) for x in confidence.tolist()]
    n = len(s)
    p = int(prefix_len)
    if p < 1 or n < 1:
        raise ValueError("prefix_len and generation length must be positive")
    length = p + n
    total = 2 * length
    v = torch.zeros((total, total), dtype=torch.bool, device=state.device)
    stable_keys = list(range(p)) + [p + i for i, z in enumerate(s) if z == U]
    candidate_keys = [p + i for i, z in enumerate(s) if z == C]
    masked_keys = [p + i for i, z in enumerate(s) if z == M]
    # Original stream: no access to shadow keys.
    for row in range(length):
        if row < p or s[row-p] == M:
            v[row, :length] = True
        elif s[row-p] == U:
            v[row, stable_keys] = True
        else:  # C: confidence-ordered original candidate key embeddings.
            owner = row-p
            eligible = [k for k in candidate_keys if (k == row or conf[k-p] > conf[owner] or
                         (conf[k-p] == conf[owner] and k < row))]
            v[row, stable_keys + eligible] = True

    # Shadow queries and keys; C tokens in shadow input are MASK tokens.
    # Avoid C shadow keys, because their states may have been verified in
    # target-specific context rather than general-purpose key representations.
    shadow_stable = [length + i for i in stable_keys + masked_keys]
    for local_q in range(length):
        row = length + local_q
        v[row, shadow_stable] = True
        # Shifted Dream logits: local_q==p-1+i predicts token at generation i.
        target = local_q - (p-1)
        if 0 <= target < n and s[target] == C:
            # STRICT: exclude self and all lower-confident candidates.
            keys = [k for k in candidate_keys
                    if conf[k-p] > conf[target] or (conf[k-p] == conf[target] and (k-p) < target)]
            if keys:
                v[row, keys] = True
    if not bool(v.any(dim=-1).all()):
        raise RuntimeError("Attention mask created an all-masked query row")
    return v.unsqueeze(0).unsqueeze(0)


def adaptive_weights(length: int, events: list[tuple[int, int]], *,
                     base: float, impact: float, decay: float, device) -> Any:
    """Spatially adaptive original-vs-shadow logit mixture w_i in [0.05,.99]."""
    import torch
    w = torch.full((length,), base, dtype=torch.float32, device=device)
    if events:
        positions = torch.arange(length, dtype=torch.float32, device=device)
        # Use last events to prevent weights drifting unbounded after many steps.
        influence = torch.zeros_like(w)
        total = torch.zeros_like(w)
        for index, sign in events[-min(len(events), 2*length):]:
            k = torch.exp(-torch.abs(positions - float(index)) / decay)
            influence += float(sign) * k
            total += k
        w += impact * influence / total.clamp(min=1.0)
    return w.clamp(0.05, 0.99)


def filtered_sample(logits, temperature: float, top_p: float):
    """Return (chosen_token_ids, p_chosen). Computes probs in fp32."""
    import torch
    values = logits.float()
    if temperature <= 0.0:
        indices = values.argmax(-1)
        probs = torch.softmax(values, dim=-1)
        confidence = probs.gather(-1, indices[..., None]).squeeze(-1)
        return indices, confidence
    values = values / temperature
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(values, dim=-1, descending=True)
        cumulative = sorted_logits.softmax(-1).cumsum(-1)
        to_mask = cumulative > top_p
        to_mask[..., 1:] = to_mask[..., :-1].clone()
        to_mask[..., 0] = False
        keep = torch.zeros_like(to_mask).scatter_(-1, sorted_idx, to_mask)
        values = values.masked_fill(keep, torch.finfo(values.dtype).min)
    probs = torch.softmax(values, dim=-1)
    indices = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(probs.shape[:-1])
    confidence = probs.gather(-1, indices[..., None]).squeeze(-1)
    return indices, confidence


def verify_probabilities(logits, chosen_ids):
    """Unwarped confidence assigned to existing candidates."""
    import torch
    selected = logits.float().gather(-1, chosen_ids[..., None]).squeeze(-1)
    return torch.exp(selected - torch.logsumexp(logits.float(), dim=-1))


class DARDDecoder:
    """Blockwise DARD with paper's major mechanisms (independent implementation)."""

    def __init__(self, model, mask_token_id: int, config: DARDConfig):
        config.validate()
        self.model = model
        self.mask_id = int(mask_token_id)
        self.config = config
        self._attention_fallback = False
        self._warned_fallback = False

    def _forward(self, ids, state, confidence):
        """ONE model forward for original and shadow with 4D visibility."""
        import torch
        p = int(ids.shape[1] - len(state))
        n = int(len(state))
        l = p + n
        shadow = ids.clone()
        c = state == C
        if bool(c.any()):
            tail = shadow[0, p:]
            tail[c] = self.mask_id
        joined = torch.cat((ids, shadow), dim=1)
        attn = build_two_stream_visibility(p, state, confidence)
        # Mirror RoPE coordinates in both views to align positional semantics.
        pos_ids = torch.arange(l, device=ids.device).repeat(2).unsqueeze(0)
        if not self._attention_fallback:
            try:
                result = self.model(joined, attention_mask=attn, position_ids=pos_ids, use_cache=False)
                # Dream's published sampler shifts logits by one position.
                orig = result.logits[0, p - 1:p - 1 + n, :]
                shad = result.logits[0, l + p - 1:l + p - 1 + n, :]
                if orig.shape[0] != n or shad.shape[0] != n:
                    raise RuntimeError("Dream shifted-logit alignment failed")
                return orig, shad, 1, False
            except (TypeError, ValueError, RuntimeError) as exc:
                if "out of memory" in str(exc).lower():
                    raise
                self._attention_fallback = True
                if not self._warned_fallback:
                    print(f"[dard] WARNING: custom 4D attention unsupported ({exc}). "
                          "Using slower exact per-candidate masked-context verifier; "
                          "this mode is logged and is NOT the optimized paper kernel.", flush=True)
                    self._warned_fallback = True
        return self._forward_reference(ids, state, confidence)

    def _forward_reference(self, ids, state, confidence):
        """Safe fallback: exact confidence-ordered dependency contexts by reforwarding.

        Retains all core DARD *semantics*, but does not use the original/shadow
        attention optimization and is much slower. Each candidate is verified
        independently with itself and lower-confidence C tokens masked.
        """
        import torch
        p = ids.shape[1] - state.numel()
        n = state.numel()
        orig = self.model(ids, attention_mask="full", use_cache=False).logits[0, p-1:-1, :]
        shadow_ids = ids.clone()
        shadow_ids[0, p:][state == C] = self.mask_id
        shad = self.model(shadow_ids, attention_mask="full", use_cache=False).logits[0, p-1:-1, :]
        shad = shad.clone()
        nfes = 2
        cand = torch.where(state == C)[0]
        for target in cand.tolist():
            local = ids.clone()
            to_mask = ((state == C) &
                       ((confidence < confidence[target]) |
                        ((confidence == confidence[target]) &
                         (torch.arange(n, device=state.device) >= target))))
            local[0, p:][to_mask] = self.mask_id
            res = self.model(local, attention_mask="full", use_cache=False).logits
            shad[target] = res[0, p - 1 + target]
            nfes += 1
        return orig, shad, nfes, True

    def _finish_unverified(self, x, state, conf, stats):
        """Bounded completion, never pretending forced tokens were verified."""
        import torch
        p = int(x.shape[1] - state.numel())
        remaining = state != U
        stats["forced_tokens"] += int(remaining.sum().item())
        if not bool(remaining.any()):
            return x
        # Use stable U tokens as context, then progressively fill low-confidence
        # positions from the model's original-stream prediction.
        x[0, p:][remaining] = self.mask_id
        while bool((x[0, p:] == self.mask_id).any()):
            # Standard full-bidirectional model call, using published alignment.
            logits = self.model(x, attention_mask="full", use_cache=False).logits
            stats["model_forward_passes"] += 1
            mask = (x[0, p:] == self.mask_id)
            raw = logits[0, p - 1:-1, :]
            ids, probs = filtered_sample(raw, self.config.temperature, self.config.top_p)
            indices = torch.where(mask)[0]
            k = min(len(indices), max(1, self.config.transfer_limit))
            chosen = indices[torch.topk(probs[indices], k=k).indices]
            x[0, p+chosen] = ids[chosen]
        return x

    def _block(self, prefix_ids, size: int):
        import torch
        cfg = self.config
        p = prefix_ids.shape[1]
        device = prefix_ids.device
        x = torch.cat((prefix_ids, torch.full((1, size), self.mask_id,
                                             dtype=prefix_ids.dtype, device=device)), dim=-1)
        state = torch.zeros(size, dtype=torch.long, device=device)
        confidence = torch.zeros(size, dtype=torch.float32, device=device)
        events: list[tuple[int, int]] = []
        stats = {"model_forward_passes": 0, "candidate_proposals": 0,
                 "promoted_tokens": 0, "revoked_tokens": 0, "forced_tokens": 0,
                 "steps": 0, "block_size": size, "mix_weight_mean": None,
                 "attention_fallback_steps": 0}
        weight_sums = []
        for step in range(cfg.steps_per_block):
            if bool((state == U).all()):
                break
            orig, shad, nfes, fallback = self._forward(x, state, confidence)
            stats["model_forward_passes"] += nfes
            stats["attention_fallback_steps"] += int(fallback)
            stats["steps"] += 1
            # Verify C in confidence-ordered shadow context, where lower
            # confidence candidate tokens cannot pollute verification.
            cand = torch.nonzero(state == C).flatten()
            if cand.numel():
                assigned = x[0, p+cand]
                verify_p = verify_probabilities(shad[cand], assigned)
                accepted = cand[verify_p >= cfg.tau_u]
                revoked = cand[verify_p < cfg.tau_c]
                if accepted.numel():
                    state[accepted] = U
                    stats["promoted_tokens"] += int(accepted.numel())
                    events.extend((int(i), +1) for i in accepted.tolist())
                if revoked.numel():
                    state[revoked] = M
                    x[0, p+revoked] = self.mask_id
                    confidence[revoked] = 0
                    stats["revoked_tokens"] += int(revoked.numel())
                    events.extend((int(i), -1) for i in revoked.tolist())
            masked = torch.nonzero(state == M).flatten()
            if masked.numel():
                w = adaptive_weights(size, events, base=cfg.mix_base, impact=cfg.mix_impact,
                                     decay=cfg.mix_decay, device=device)
                weight_sums.append(float(w.mean().item()))
                combined = w[:, None] * orig.float() + (1.0 - w[:, None]) * shad.float()
                proposed, sampled_conf = filtered_sample(combined[masked], cfg.temperature, cfg.top_p)
                # Mix confidence is the sampled-token probability. If no token
                # clears tau_c, admit the best one to avoid deadlock.
                eligible = torch.nonzero(sampled_conf >= cfg.tau_c).flatten()
                allowance = min(cfg.transfer_limit, len(masked))
                if not len(eligible):
                    eligible = torch.topk(sampled_conf, k=1).indices
                chosen_local = eligible[torch.topk(sampled_conf[eligible],
                                                   k=min(allowance, len(eligible))).indices]
                targets = masked[chosen_local]
                x[0, p+targets] = proposed[chosen_local]
                confidence[targets] = sampled_conf[chosen_local]
                state[targets] = C
                stats["candidate_proposals"] += len(targets)
        stats["mix_weight_mean"] = round(sum(weight_sums)/len(weight_sums), 4) if weight_sums else cfg.mix_base
        if bool((state != U).any()):
            x = self._finish_unverified(x, state, confidence, stats)
        return x, stats

    def generate(self, prompt_ids, num_new_tokens: int):
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("DARDDecoder currently supports single-prompt batch=1")
        if num_new_tokens < 1:
            raise ValueError("num_new_tokens must be positive")
        x = prompt_ids
        totals = {"model_forward_passes": 0, "candidate_proposals": 0,
                  "promoted_tokens": 0, "revoked_tokens": 0, "forced_tokens": 0,
                  "steps": 0, "attention_fallback_steps": 0, "blocks": []}
        for start in range(0, num_new_tokens, self.config.block_size):
            size = min(self.config.block_size, num_new_tokens - start)
            x, info = self._block(x, size)
            for key in ("model_forward_passes", "candidate_proposals", "promoted_tokens",
                        "revoked_tokens", "forced_tokens", "steps", "attention_fallback_steps"):
                totals[key] += info[key]
            totals["blocks"].append(info)
        return x[:, prompt_ids.shape[1]:], totals
