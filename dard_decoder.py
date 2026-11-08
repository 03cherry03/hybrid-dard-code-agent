"""DARD core ideas adapted to Dream's shifted-logit diffusion backbone.

Uses original and fully masked shadow context keys, confidence-ordered candidate
verification, M/C/U revocation, and the paper's prior-ratio logit mixing.
Separate predictor queries preserve Dream's t-1 residual/position alignment.
This shifted-backbone adapter is not author code or an exact LLaDA reproduction.
Native all-masked parity is tested; performance must be measured on real models.
Reference: https://arxiv.org/html/2608.26574v1, section 3 and Algorithm 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any
import math
import inspect
import time

from dard_attention import build_two_stream_visibility, packed_inputs, compact_packed_inputs


@dataclass(frozen=True)
class DARDConfig:
    block_size: int = 128
    steps_per_block: int = 64
    tau_c: float = 0.50
    tau_u: float = 0.80
    transfer_limit: int = 0  # 0 commits every position passing the confidence threshold.
    mix_prior: float = 0.10
    mix_decay: float = 8.0
    temperature: float = 0.3
    top_p: float = 0.95
    allow_attention_fallback: bool = False
    diagnostics: str = "summary"
    max_step_records: int = 256
    cycle_window: int = 8
    compact_queries: bool = True
    rectangular_attention: bool = True
    completion_policy: str = "legacy_force"
    completion_verification_steps: int = 2

    def validate(self) -> None:
        if not (1 <= self.block_size <= 1024 and 1 <= self.steps_per_block <= 4096):
            raise ValueError("block_size and steps_per_block are out of range")
        if not (0 < self.tau_c <= self.tau_u < 1):
            raise ValueError("Require 0 < tau_c <= tau_u < 1")
        if isinstance(self.transfer_limit, bool) or not isinstance(self.transfer_limit, int) or self.transfer_limit < 0:
            raise ValueError("transfer_limit must be a nonnegative integer (0 means uncapped)")
        if not (0 < self.mix_prior and math.isfinite(self.mix_prior) and self.mix_decay > 0):
            raise ValueError("Invalid adaptive-mixing parameters")
        if not (self.temperature >= 0 and 0 < self.top_p <= 1):
            raise ValueError("Invalid sampling parameters")
        if not all(math.isfinite(v) for v in (self.mix_decay, self.temperature)):
            raise ValueError("Sampling and mixing parameters must be finite")
        if isinstance(self.cycle_window, bool) or not isinstance(self.cycle_window, int) or not 0 <= self.cycle_window <= 64:
            raise ValueError("cycle_window must be an integer between 0 and 64")
        if not isinstance(self.allow_attention_fallback, bool):
            raise ValueError("allow_attention_fallback must be boolean")
        if not isinstance(self.compact_queries, bool) or not isinstance(self.rectangular_attention, bool):
            raise ValueError("compact_queries and rectangular_attention must be boolean")
        if self.completion_policy not in ("legacy_force", "verified_only", "force_then_verify",
                                          "resample_then_verify"):
            raise ValueError("Unknown completion_policy")
        if (isinstance(self.completion_verification_steps, bool)
                or not isinstance(self.completion_verification_steps, int)
                or not 0 <= self.completion_verification_steps <= 32):
            raise ValueError("completion_verification_steps must be an integer between 0 and 32")
        if self.completion_policy in ("force_then_verify", "resample_then_verify") and self.completion_verification_steps < 1:
            raise ValueError(f"{self.completion_policy} requires at least one completion verification step")
        if self.diagnostics not in ("summary", "steps", "profile"):
            raise ValueError("diagnostics must be summary, steps, or profile")
        if (isinstance(self.max_step_records, bool) or not isinstance(self.max_step_records, int)
                or not 0 <= self.max_step_records <= 100000):
            raise ValueError("max_step_records must be an integer between 0 and 100000")


M, C, U = 0, 1, 2


def _verification_context(state, confidence):
    """Allowed block keys for each target, including harmless masked keys."""
    import torch
    n = state.numel()
    positions = torch.arange(n, device=state.device)
    higher = ((confidence[None, :] > confidence[:, None]) |
              ((confidence[None, :] == confidence[:, None]) &
               (positions[None, :] < positions[:, None])))
    allowed = ((state == U) | (state == M))[None, :].expand(n, n).clone()
    allowed |= (state == C)[:, None] & (state == C)[None, :] & higher
    allowed[positions, positions] = state == M
    return allowed


def adaptive_weights(length: int, events: list[tuple[int, int]], *,
                     prior: float, decay: float, device) -> Any:
    """Paper Eq. 5-6: (P + prior)/(P + D + prior), half-life ``decay``.

    Only current-step candidate promotions/demotions contribute. With no
    demotions, original-context weight is exactly one, as in the paper.
    """
    import torch
    if not (prior > 0 and math.isfinite(prior) and decay > 0 and math.isfinite(decay)):
        raise ValueError("Mixing prior and decay must be positive finite values")
    if not events:
        return torch.ones(length, dtype=torch.float32, device=device)
    event = torch.tensor(events, dtype=torch.float32, device=device)
    positions = torch.arange(length, dtype=torch.float32, device=device)
    kernel = torch.exp2(-torch.abs(positions[:, None] - event[None, :, 0]) / decay)
    promoted = (kernel * (event[None, :, 1] > 0)).sum(-1)
    demoted = (kernel * (event[None, :, 1] < 0)).sum(-1)
    return (promoted + prior) / (promoted + demoted + prior)


def _validate_logits(logits):
    import torch
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("Expected logits with at least two vocabulary entries")
    if bool((torch.isnan(logits) | torch.isposinf(logits)).any()):
        raise ValueError("Model returned NaN or positive-infinite logits")
    if not bool(torch.isfinite(logits).any(-1).all()):
        raise ValueError("Model returned a row with no finite token logits")


def filtered_sample(logits, temperature: float, top_p: float, *, forbidden_token_ids=()):
    """Sample with temperature/top-p; confidence always uses UNWARPED logits.

    Forbidden IDs affect sampling only. Confidence retains the original model
    probability, including any mass originally assigned to the mask token.
    """
    import torch
    if not (temperature >= 0 and math.isfinite(temperature) and 0 < top_p <= 1):
        raise ValueError("Invalid sampling parameters")
    raw = logits.float()
    _validate_logits(raw)
    values = raw.clone()
    for token_id in forbidden_token_ids:
        if not 0 <= token_id < values.shape[-1]:
            raise ValueError("Forbidden token id is outside the model vocabulary")
        values[..., token_id] = -torch.inf
    if not bool(torch.isfinite(values).any(-1).all()):
        raise ValueError("No finite non-forbidden token is available")
    if temperature <= 0:
        indices = values.argmax(-1)
    else:
        values = values / temperature
        if top_p < 1:
            sorted_logits, sorted_idx = torch.sort(values, dim=-1, descending=True)
            remove = sorted_logits.softmax(-1).cumsum(-1) > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            remove = torch.zeros_like(remove).scatter_(-1, sorted_idx, remove)
            values = values.masked_fill(remove, -torch.inf)
        probs = torch.softmax(values, dim=-1)
        indices = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(probs.shape[:-1])
    # ``raw`` was validated above and only its clone is warped. Revalidating
    # the full vocabulary here adds two GPU-to-host synchronization points on
    # every proposal/forced-fill call without checking any new values.
    return indices, _probabilities_from_validated_logits(raw, indices)


def _probabilities_from_validated_logits(raw, chosen_ids):
    import torch
    selected = raw.gather(-1, chosen_ids[..., None]).squeeze(-1)
    return torch.exp(selected - torch.logsumexp(raw, dim=-1))


def verify_probabilities(logits, chosen_ids):
    """Unwarped probability for either existing or newly sampled token IDs."""
    import torch
    raw = logits.float()
    _validate_logits(raw)
    return _probabilities_from_validated_logits(raw, chosen_ids)


def _mix_masked_logits(original, shadow, weights, masked):
    """Mix full or compact original rows in masked-index order using FP32."""
    import torch
    selected_weights = weights[masked, None]
    if original.shape[0] == shadow.shape[0]:
        original_rows = original[masked]
    elif original.shape[0] == masked.numel():
        original_rows = original
    else:
        raise ValueError("Original logits must contain all rows or exactly the masked rows")
    original_term = selected_weights * original_rows.float()
    shadow_term = (1-selected_weights) * shadow[masked].float()
    # Exact endpoint weights are meaningful in Eq. 6; 0 * -inf must not
    # introduce NaNs for vocabulary entries masked by a model/sampler.
    return (torch.where(selected_weights == 0, 0.0, original_term)
            + torch.where(selected_weights == 1, 0.0, shadow_term))


class DARDDecoder:
    """Bounded single-prompt research decoder with explicit diagnostic fallback."""

    def __init__(self, model, mask_token_id: int, config: DARDConfig, *, eos_token_ids=()):
        config.validate()
        self.model = model
        self.mask_id = int(mask_token_id)
        if self.mask_id < 0:
            raise ValueError("mask_token_id must be nonnegative")
        self.config = config
        # Dream exposes this parameter explicitly. A generic **kwargs is not
        # evidence that a different model implements tail-only projection.
        try:
            forward = getattr(model, "forward", None) or getattr(model, "__call__", None)
            self._tail_logits = "num_logits_to_keep" in inspect.signature(forward).parameters
        except (TypeError, ValueError):
            self._tail_logits = False
        self.logit_projection_policy = ("needed_tail_rows_v1" if self._tail_logits
                                        else "full_sequence_compatibility")
        self.eos_token_ids = tuple(sorted(set(int(i) for i in eos_token_ids)))
        if any(i < 0 or i == self.mask_id for i in self.eos_token_ids):
            raise ValueError("Terminal IDs must be nonnegative and different from MASK")
        self._attention_fallback = False
        self._warned_fallback = False
        self._current_block_diagnostics = None
        self._current_block_stats = None
        self._forced_record_count = 0
        self._pending_forced_records = []
        self.last_diagnostics = self._new_diagnostics()

    def _new_diagnostics(self):
        return {"schema_version": 1, "mode": self.config.diagnostics,
                "instrumentation_affects_timing": self.config.diagnostics != "summary",
                "complete": False, "partial": True, "phase": "initializing",
                "events": [], "records_dropped": 0, "trace_truncated": False,
                "max_step_records": self.config.max_step_records,
                "completion_policy": self.config.completion_policy,
                "completion_verification_budget": self.config.completion_verification_steps,
                "completion_verification_steps": 0, "completion_verification_forward_passes": 0,
                "completion_reverified_tokens": 0, "completion_rejected_tokens": 0,
                "completion_remasked_tokens": 0, "incomplete": True,
                "verification_certified": False, "unresolved_mask_tokens": None,
                "completion_unverified_tokens": None, "termination_cause": "running",
                "model_forward_attempts": 0, "model_forward_succeeded": 0,
                "model_forward_failed": 0, "component_wall_s": {},
                "logit_projection_policy": self.logit_projection_policy,
                "projected_logit_rows": 0, "native_all_masked_forwards": 0,
                "compact_queries": self.config.compact_queries,
                "rectangular_attention": self.config.rectangular_attention and bool(
                    getattr(self.model, "_dard_rectangular_attention", False)),
                "packed_forwards": 0, "packed_query_rows": 0, "packed_key_rows": 0,
                "omitted_original_query_rows": 0, "max_actual_input_length": 0,
                "attention_fallback_used": self._attention_fallback,
                "timing_semantics": "not_measured", "blocks": [],
                "notes": ["Summary adds no model calls or RNG draws; at most 512 forced-token records are serialized once per generation and included in latency.",
                          "Steps/profile add tensor summaries and perturb measured latency.",
                          "Profile uses synchronized wall time, not isolated GPU kernel time.",
                          "Component timings exclude trace summaries and loop/stop-check overhead.",
                          "Stalled steps mean no increase in verified U tokens, including fresh proposals.",
                          "Confidence is a decoder signal, not a correctness estimate.",
                          "Completion certification is separate from functional correctness; incomplete means the selected completion policy was not satisfied.",
                          "Completion rejection counts are token-verification events, including repeated checks, not unique token positions."]}

    @contextmanager
    def _measure(self, component, device):
        """Opt-in synchronized component wall timings; no clocks/sync by default."""
        if self.config.diagnostics != "profile":
            yield
            return
        import torch
        is_cuda = device.type == "cuda"
        self.last_diagnostics["timing_semantics"] = (
            "cuda_device_synchronized_wall" if is_cuda else "cpu_wall")
        if is_cuda:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        body_failed = False
        try:
            yield
        except BaseException:
            body_failed = True
            raise
        finally:
            # A failed CUDA operation must retain the original exception and the
            # already available diagnostics even if synchronization also fails.
            sync_error = None
            try:
                if is_cuda:
                    torch.cuda.synchronize(device)
            except Exception as exc:
                sync_error = exc
                self.last_diagnostics["timing_sync_error"] = type(exc).__name__
            elapsed = time.perf_counter() - started
            for owner in (self.last_diagnostics, self._current_block_diagnostics):
                if owner is not None:
                    times = owner["component_wall_s"]
                    times[component] = times.get(component, 0.0) + elapsed
            if sync_error is not None and not body_failed:
                raise sync_error

    def _call_model(self, ids, **kwargs):
        """Count attempts before invocation, including failed compatibility probes."""
        self.last_diagnostics["phase"] = "model_forward"
        owners = (self.last_diagnostics, self._current_block_diagnostics)
        invoked = False
        try:
            with self._measure("model_forward", ids.device):
                for owner in owners:
                    if owner is not None:
                        owner["model_forward_attempts"] += 1
                invoked = True
                output = self.model(ids, **kwargs)
        except Exception:
            if invoked:
                for owner in owners:
                    if owner is not None:
                        owner["model_forward_failed"] += 1
            raise
        for owner in owners:
            if owner is not None:
                owner["model_forward_succeeded"] += 1
                owner["max_actual_input_length"] = max(owner.get("max_actual_input_length", 0), int(ids.shape[1]))
                logits = getattr(output, "logits", None)
                if getattr(logits, "ndim", 0) == 3:
                    owner["projected_logit_rows"] += int(logits.shape[0] * logits.shape[1])
        return output

    def _projection_options(self, rows):
        return {"num_logits_to_keep": rows} if self._tail_logits else {}

    def _plain_block_logits(self, ids, block_size):
        """Keep the B shifted predictors plus the final unused token row."""
        logits = self._call_model(ids, attention_mask="full", use_cache=False,
                                  output_attentions=False,
                                  **self._projection_options(block_size + 1)).logits
        expected = block_size + 1 if self._tail_logits else ids.shape[1]
        if logits.ndim != 3 or logits.shape[:2] != (1, expected):
            raise RuntimeError("DARD model returned unexpected plain-logit shape")
        start = 0 if self._tail_logits else ids.shape[1] - block_size - 1
        return logits[0, start:start + block_size]

    def _event(self, event):
        if self.config.diagnostics == "summary":
            return
        diag = self.last_diagnostics
        if len(diag["events"]) < self.config.max_step_records:
            diag["events"].append(event)
        else:
            diag["records_dropped"] += 1
            diag["trace_truncated"] = True

    @staticmethod
    def _tensor_summary(values):
        """Only used in detailed modes; never retain tensors or full logits."""
        if not values.numel():
            return {"count": 0, "min": None, "mean": None, "max": None}
        return {"count": int(values.numel()), "min": float(values.min().item()),
                "mean": float(values.float().mean().item()), "max": float(values.max().item())}

    def _packed_inputs(self, ids, state, confidence):
        return packed_inputs(ids, state, confidence, self.mask_id)

    def _forward(self, ids, state, confidence):
        if self._attention_fallback:
            return self._forward_reference(ids, state, confidence)
        n = int(state.numel())
        p = int(ids.shape[1] - n)
        if bool((state == M).all()):
            # No candidate/reliable token is being verified. The two streams
            # represent the same native context and paper Eq. 6 gives weight1.
            # Use native coordinates directly: fewer rows and no packed-shape
            # BF16 perturbation to the pretrained initial predictions.
            logits = self._plain_block_logits(ids, n)
            for owner in (self.last_diagnostics, self._current_block_diagnostics):
                if owner is not None:
                    owner["native_all_masked_forwards"] = owner.get("native_all_masked_forwards", 0) + 1
            return logits, logits, 1, False
        self.last_diagnostics["phase"] = "layout_construction"
        rectangular = self.config.rectangular_attention and bool(
            getattr(self.model, "_dard_rectangular_attention", False))
        with self._measure("layout_construction", ids.device):
            if self.config.compact_queries:
                joined, positions, attention, masked = compact_packed_inputs(
                    ids, state, confidence, self.mask_id, rectangular=rectangular)
                original_rows = int(masked.numel())
            else:
                joined, positions = self._packed_inputs(ids, state, confidence)
                attention = build_two_stream_visibility(p, state, confidence)
                if rectangular:
                    attention = attention[..., :p+2*n]
                original_rows = n
        prediction_rows = original_rows + n
        for owner in (self.last_diagnostics, self._current_block_diagnostics):
            if owner is not None:
                owner["packed_forwards"] = owner.get("packed_forwards", 0) + 1
                owner["packed_query_rows"] = owner.get("packed_query_rows", 0) + int(joined.shape[1])
                owner["packed_key_rows"] = owner.get("packed_key_rows", 0) + int(attention.shape[-1])
                owner["omitted_original_query_rows"] = owner.get("omitted_original_query_rows", 0) + n-original_rows
                owner["last_packed_input_length"] = int(joined.shape[1])
                owner["last_packed_attention_mask_elements"] = int(attention.numel())
        try:
            result = self._call_model(joined, attention_mask=attention, position_ids=positions,
                                      use_cache=False, output_attentions=False,
                                      **self._projection_options(prediction_rows))
        except (TypeError, ValueError, RuntimeError) as exc:
            if getattr(exc, "abort_dard_generation", False) or "out of memory" in str(exc).lower():
                raise
            if not self.config.allow_attention_fallback:
                raise RuntimeError(
                    "DARD packed-attention forward failed. Diagnostic fallback is disabled; "
                    "verify Dream/4D-mask compatibility or explicitly enable "
                    "allow_attention_fallback for a diagnostic run. " + str(exc)) from exc
            self._attention_fallback = True
            self.last_diagnostics["attention_fallback_used"] = True
            self._event({"type": "attention_fallback", "phase": "model_forward",
                         "block_index": len(self.last_diagnostics["blocks"])-1,
                         "error_type": type(exc).__name__, "error": str(exc)[:2000]})
            if not self._warned_fallback:
                print(f"[dard] WARNING: packed-attention failed ({exc}); using isolated-context "
                      "diagnostic fallback. Its timings are not eligible for speed claims.", flush=True)
                self._warned_fallback = True
            orig, shad, calls, fallback = self._forward_reference(ids, state, confidence)
            return orig, shad, calls + 1, fallback
        expected = prediction_rows if self._tail_logits else joined.shape[1]
        if result.logits.ndim != 3 or result.logits.shape[:2] != (1, expected):
            raise RuntimeError("DARD model returned unexpected packed-logit shape")
        start = 0 if self._tail_logits else p + 2*n
        # Do not shift again: query banks ALREADY use the t-1 coordinates.
        return (result.logits[0, start:start+original_rows],
                result.logits[0, start+original_rows:start+prediction_rows], 1, False)

    def _forward_reference(self, ids, state, confidence):
        """Slow isolated-context diagnostic, not numerically equivalent to packing.

        Forbidden tokens are physically masked before full model invocations,
        so they cannot re-enter verification through another transformer layer.
        """
        import torch
        p, n = ids.shape[1] - state.numel(), state.numel()
        orig = self._plain_block_logits(ids, n)
        with self._measure("layout_construction", ids.device):
            shadow = ids.clone()
            shadow[0, p:][state == C] = self.mask_id
        shad = self._plain_block_logits(shadow, n).clone()
        calls = 2
        with self._measure("layout_construction", ids.device):
            allowed = _verification_context(state, confidence)
        for target in torch.where(state != M)[0].tolist():
            with self._measure("layout_construction", ids.device):
                local = ids.clone()
                local[0, p:] = torch.where(allowed[target], ids[0, p:], self.mask_id)
            logits = self._plain_block_logits(local, n)
            shad[target] = logits[target]
            calls += 1
        return orig, shad, calls, True

    def _terminal_cutoff(self, tokens):
        import torch
        if not self.eos_token_ids:
            return None
        terminal = torch.zeros_like(tokens, dtype=torch.bool)
        for token_id in self.eos_token_ids:
            terminal |= tokens == token_id
        found = torch.where(terminal)[0]
        return int(found[0].item()) + 1 if found.numel() else None

    def _finish_unverified(self, x, state, verified, stats, *, candidate_confidence=None,
                           trim_terminal=True):
        """Bounded completion fallback, explicitly reported separately from verification."""
        import torch
        p = int(x.shape[1] - state.numel())
        self.last_diagnostics["phase"] = "forced_completion"
        with self._measure("forced_preparation", x.device):
            remaining = ~((state == U) & verified)
            x[0, p:][remaining] = self.mask_id
            if candidate_confidence is not None:
                state[remaining] = M
                verified[remaining] = False
                candidate_confidence[remaining] = 0
            remaining_count = int(remaining.sum().item())
            completion_limit = self.config.transfer_limit or 8
            bound = math.ceil(remaining_count / completion_limit)
        block = self._current_block_diagnostics
        forced = {"reason": stats["stop_reason"], "unverified_slots": remaining_count,
                  "max_forward_calls": bound, "filled_tokens": 0, "complete": False,
                  "token_records": [], "token_records_dropped": 0,
                  "token_record_limit_per_generation": 512,
                  "output_offset": p-self.last_diagnostics.get("prompt_tokens", p),
                  "confidence_semantics": "unwarped_native_probability_before_forced_assignment"}
        block["forced_completion"] = forced
        self._event({"type": "forced_completion", "block_index": block["block_index"],
                     "phase": "begin", **forced})
        for force_step in range(bound):
            masked = torch.where(x[0, p:] == self.mask_id)[0]
            if not masked.numel():
                break
            logits = self._plain_block_logits(x, state.numel())
            stats["model_forward_passes"] += 1
            stats["forced_forward_passes"] += 1
            self.last_diagnostics["phase"] = "forced_resampling"
            with self._measure("forced_resampling", x.device):
                proposed, probs = filtered_sample(logits[masked], self.config.temperature,
                                                  self.config.top_p, forbidden_token_ids=(self.mask_id,))
                count = min(masked.numel(), completion_limit)
                selected = torch.topk(probs, k=count).indices
                x[0, p+masked[selected]] = proposed[selected]
                if candidate_confidence is not None:
                    # Forced tokens are provisional candidates, even below
                    # tau_c. None becomes certified from its sampling score.
                    state[masked[selected]] = C
                    candidate_confidence[masked[selected]] = probs[selected]
                    verified[masked[selected]] = False
                # Retain only bounded small tensors, without per-step CPU sync.
                # Serialize once at generation exit, including partial failures.
                record_count = min(count, 512-self._forced_record_count)
                if record_count:
                    retained = selected[:record_count]
                    record = torch.stack((torch.full_like(probs[retained], force_step+1, dtype=torch.float64),
                                          masked[retained].double(), proposed[retained].double(),
                                          probs[retained].double()), dim=-1)
                    self._pending_forced_records.append((forced, record))
                    self._forced_record_count += record_count
                forced["token_records_dropped"] += count-record_count
            stats["forced_tokens"] += count
            forced["filled_tokens"] += count
            self._event({"type": "forced_completion", "block_index": block["block_index"],
                         "phase": "fill", "force_step": force_step + 1, "filled_tokens": count,
                         "remaining_slots": int(masked.numel()) - count,
                         "input_length": int(x.shape[1]), "forward_calls": 1,
                         "confidence": self._tensor_summary(probs[selected])
                         if self.config.diagnostics != "summary" else None})
            cutoff = self._terminal_cutoff(x[0, p:])
            if cutoff is not None and not bool((x[0, p:p+cutoff] == self.mask_id).any()):
                forced.update(complete=True, stop_reason="eos")
                return x[:, :p+cutoff] if trim_terminal else x
        if bool((x[0, p:] == self.mask_id).any()):
            raise RuntimeError("Bounded DARD completion failed to fill every output slot")
        forced.update(complete=True, stop_reason="filled")
        return x

    def _completion_status(self, x, state, verified, prefix_length):
        """Certification concerns the emitted prefix, including its terminal.

        Incomplete policies retain the original canvas; this count still reports
        MASKs before the first terminal, so downstream text cleanup cannot hide
        an unresolved output token from the result metadata.
        """
        import torch
        tokens = x[0, prefix_length:]
        cutoff = self._terminal_cutoff(tokens)
        extent = cutoff if cutoff is not None else int(tokens.numel())
        certified = (state[:extent] == U) & verified[:extent]
        return {"verification_certified": bool(certified.all()),
                "completion_unverified_tokens": int((~certified).sum().item()),
                "unresolved_mask_tokens": int((tokens[:extent] == self.mask_id).sum().item())}

    def _verify_completion(self, x, state, confidence, verified, stats, prefix_length):
        """Bounded DARD verification after forcing; never resample a rejection.

        A rejection is left masked and explicitly incomplete. Calls use the same
        model entry point as ordinary decoding, including caller deadline and
        forward-budget hooks. This is a verification extension, not a claim of
        the original paper's stopping policy.
        """
        import torch
        block = self._current_block_diagnostics
        for step in range(self.config.completion_verification_steps):
            if self._completion_status(x, state, verified, prefix_length)["verification_certified"]:
                break
            decoded = torch.where(state != M)[0]
            if not decoded.numel():
                break
            old_state = state.clone()
            tokens = x[0, prefix_length+decoded].clone()
            _, shadow, calls, fallback = self._forward(x, state, confidence)
            stats["model_forward_passes"] += calls
            stats["completion_verification_forward_passes"] += calls
            self.last_diagnostics["completion_verification_forward_passes"] += calls
            stats["attention_fallback_steps"] += int(fallback)
            self.last_diagnostics["phase"] = "completion_verification"
            with self._measure("completion_verification", x.device):
                probability = verify_probabilities(shadow[decoded], tokens)
                updated = torch.full_like(decoded, M)
                updated[probability > self.config.tau_c] = C
                updated[probability > self.config.tau_u] = U
                state[decoded] = updated
                confidence[decoded] = probability
                verified[decoded] = updated == U
                rejected = updated != U
                remasked = decoded[updated == M]
                x[0, prefix_length+remasked] = self.mask_id
                confidence[remasked] = 0
                progress = {"completion_verification_steps": 1,
                            "completion_reverified_tokens": int(decoded.numel()),
                            "completion_rejected_tokens": int(rejected.sum().item()),
                            "completion_remasked_tokens": int(remasked.numel())}
                for owner in (stats, self.last_diagnostics):
                    for key, value in progress.items():
                        owner[key] += value
                stats["u_reverified_tokens"] += int((old_state[decoded] == U).sum().item())
                stats["promoted_tokens"] += int(((old_state[decoded] == C) & (updated == U)).sum().item())
                stats["revoked_tokens"] += int(remasked.numel())
                stats["demoted_tokens"] += int(((old_state[decoded] == U) & (updated == C)).sum().item())
            block["completion_state_counts"] = {name: int((state == value).sum().item())
                                                for name, value in (("M", M), ("C", C), ("U", U))}
            block["completion_verified_u_tokens"] = int(((state == U) & verified).sum().item())
            if self.config.diagnostics != "summary":
                limit = min(64, int(decoded.numel()))
                records = torch.stack((decoded[:limit].double(), tokens[:limit].double(),
                                       probability[:limit].double()), dim=-1).cpu().tolist()
                self._event({"type": "completion_verification", "block_index": block["block_index"],
                    "step": step+1, "forward_calls": calls,
                    "state_after": dict(block["completion_state_counts"]),
                    "token_records": [{"block_position": int(position),
                        "output_position": prefix_length-self.last_diagnostics.get("prompt_tokens", prefix_length)+int(position),
                        "token_id": int(token), "probability": prob,
                        "verified_u": prob > self.config.tau_u, "remasked": prob <= self.config.tau_c}
                        for position, token, prob in records],
                    "token_records_dropped": int(decoded.numel())-limit})
        return x

    def _resample_then_verify(self, x, state, confidence, verified, stats, prefix_length):
        """Repair cycle/fixed-point tails by remasking and resampling them.

        ``legacy_force`` assigns provisional tokens to unresolved positions and
        marks the canvas complete.  This policy keeps those positions masked,
        samples them from a fresh native masked forward, then verifies them
        through the normal DARD shadow stream.  A rejected position is remasked
        for the next bounded attempt; it is never counted as certified merely
        because the budget expired.
        """
        import torch
        block = self._current_block_diagnostics
        p = int(prefix_length)
        n = int(state.numel())
        budget = self.config.completion_verification_steps
        for step in range(budget):
            status = self._completion_status(x, state, verified, p)
            if status["verification_certified"]:
                break
            cutoff = self._terminal_cutoff(x[0, p:])
            extent = cutoff if cutoff is not None else n
            certified = (state[:extent] == U) & verified[:extent]
            unresolved = torch.where(~certified)[0]
            if not unresolved.numel():
                break
            # Start every retry from a clean masked canvas for only the
            # unverified positions. Verified tokens remain fixed.
            x[0, p+unresolved] = self.mask_id
            state[unresolved] = M
            confidence[unresolved] = 0
            verified[unresolved] = False
            with self._measure("completion_resampling", x.device):
                logits = self._plain_block_logits(x, n)
                stats["model_forward_passes"] += 1
                stats["completion_verification_forward_passes"] += 1
                self.last_diagnostics["completion_verification_forward_passes"] += 1
                proposed, probabilities = filtered_sample(
                    logits[unresolved], self.config.temperature, self.config.top_p,
                    forbidden_token_ids=(self.mask_id,))
                x[0, p+unresolved] = proposed
                state[unresolved] = C
                confidence[unresolved] = probabilities
                stats["candidate_proposals"] += int(unresolved.numel())
            # Verify the new proposals with the same shadow stream used by the
            # ordinary DARD loop.  This forward is intentionally separate from
            # proposal generation so its cost is visible in diagnostics.
            _, shadow, calls, fallback = self._forward(x, state, confidence)
            stats["model_forward_passes"] += calls
            stats["completion_verification_forward_passes"] += calls
            self.last_diagnostics["completion_verification_forward_passes"] += calls
            stats["attention_fallback_steps"] += int(fallback)
            with self._measure("completion_verification", x.device):
                probability = verify_probabilities(shadow[unresolved], x[0, p+unresolved])
                updated = torch.full_like(unresolved, M)
                updated[probability > self.config.tau_c] = C
                updated[probability > self.config.tau_u] = U
                state[unresolved] = updated
                confidence[unresolved] = probability
                verified[unresolved] = updated == U
                rejected = updated != U
                remasked = unresolved[updated == M]
                x[0, p+remasked] = self.mask_id
                confidence[remasked] = 0
                stats["completion_verification_steps"] += 1
                stats["completion_reverified_tokens"] += int(unresolved.numel())
                stats["completion_rejected_tokens"] += int(rejected.sum().item())
                stats["completion_remasked_tokens"] += int(remasked.numel())
                stats["u_reverified_tokens"] += int((updated == U).sum().item())
                stats["promoted_tokens"] += int((updated == U).sum().item())
                stats["revoked_tokens"] += int(remasked.numel())
            block["completion_state_counts"] = {
                name: int((state == value).sum().item())
                for name, value in (("M", M), ("C", C), ("U", U))}
            block["completion_verified_u_tokens"] = int(((state == U) & verified).sum().item())
            self._event({"type": "completion_resample_verify", "block_index": block["block_index"],
                         "step": step + 1, "resampled_tokens": int(unresolved.numel()),
                         "rejected_tokens": int(rejected.sum().item()),
                         "remasked_tokens": int(remasked.numel()), "forward_calls": 1 + calls})
        return x

    def _serialize_forced_records(self):
        """One bounded GPU-to-host transfer after generation for forced tokens."""
        import torch
        if not self._pending_forced_records:
            return
        owners, pieces = zip(*self._pending_forced_records)
        records = torch.cat(pieces, dim=0).detach().cpu().tolist()
        offset = 0
        for owner, piece in zip(owners, pieces):
            for step, position, token, probability in records[offset:offset+piece.shape[0]]:
                owner["token_records"].append({
                    "force_step": int(step), "block_position": int(position),
                    "output_position": owner["output_offset"]+int(position),
                    "token_id": int(token), "probability": float(probability)})
            offset += piece.shape[0]
        self._pending_forced_records.clear()

    def _block(self, prefix_ids, size: int):
        import torch
        cfg = self.config
        self.last_diagnostics["phase"] = "block_initialization"
        p, device = prefix_ids.shape[1], prefix_ids.device
        x = torch.cat((prefix_ids, torch.full((1, size), self.mask_id,
                                             dtype=prefix_ids.dtype, device=device)), dim=-1)
        state = torch.zeros(size, dtype=torch.long, device=device)
        confidence = torch.zeros(size, dtype=torch.float32, device=device)
        verified = torch.zeros(size, dtype=torch.bool, device=device)
        stats = {"model_forward_passes": 0, "candidate_proposals": 0,
                 "promoted_tokens": 0, "revoked_tokens": 0, "forced_tokens": 0,
                 "demoted_tokens": 0, "u_reverified_tokens": 0, "forced_forward_passes": 0,
                 "steps": 0, "block_size": size, "mix_weight_mean": None,
                 "attention_fallback_steps": 0, "stop_reason": "step_budget",
                 "completion_policy": cfg.completion_policy,
                 "completion_verification_budget": cfg.completion_verification_steps,
                 "completion_verification_steps": 0, "completion_verification_forward_passes": 0,
                 "completion_reverified_tokens": 0, "completion_rejected_tokens": 0,
                 "completion_remasked_tokens": 0, "incomplete": False}
        self._current_block_stats = stats
        block = {"block_index": len(self.last_diagnostics["blocks"]),
                 "completion_policy": cfg.completion_policy,
                 "completion_verification_budget": cfg.completion_verification_steps,
                 "incomplete": True, "verification_certified": False,
                 "unresolved_mask_tokens": None, "termination_cause": "running",
                 "prefix_length": int(p), "block_size": size,
                 "plain_input_length": int(p + size), "packed_input_length": int(p + 4*size),
                 "packed_attention_mask_elements": int((p + 4*size)**2),
                 "packed_attention_mask_bool_bytes": int((p + 4*size)**2),
                 "packed_dimensions_scope": "uncompacted_upper_bound; actual sizes are recorded separately",
                 "mask_bytes_scope": "single_explicit_boolean_mask_excludes_model_attention_workspace",
                 "thresholds": {"tau_c": cfg.tau_c, "tau_u": cfg.tau_u,
                                "comparison": "strict_greater_than"},
                 "model_forward_attempts": 0, "model_forward_succeeded": 0,
                 "model_forward_failed": 0, "component_wall_s": {},
                 "logit_projection_policy": self.logit_projection_policy,
                 "projected_logit_rows": 0, "native_all_masked_forwards": 0,
                 "steps": 0, "last_completed_step": 0,
                 "state_counts_scope": "last_completed_verification_step_before_any_forced_fill",
                 "stalled_steps": 0, "unchanged_state_steps": 0,
                 "negative_verified_progress_steps": 0, "max_stall_run": 0,
                 "zero_eligible_steps": 0, "revisited_tokens": 0 if cfg.diagnostics != "summary" else None,
                 "max_position_visits": 0 if cfg.diagnostics != "summary" else None,
                 "state_counts": {"M": size, "C": 0, "U": 0},
                 "verified_u_tokens": 0, "forced_completion": None, "complete": False,
                 "stop_reason": "running"}
        self.last_diagnostics["blocks"].append(block)
        self._current_block_diagnostics = block
        visits = [0] * size if cfg.diagnostics != "summary" else None
        stall_run = 0
        weight_sums = []
        history = []
        for step in range(cfg.steps_per_block):
            if cfg.temperature == 0 and cfg.cycle_window:
                # Retain exact IDs, state flags and float32 confidence bits;
                # equality is a deterministic state proof, not a lossy hash.
                signature = torch.cat((x[0, p:].long(), state,
                                       confidence.view(torch.int32).long(), verified.long()))
                previous = next((old_step for old_step, old in history if torch.equal(signature, old)), None)
                if previous is not None:
                    stats["stop_reason"] = "deterministic_cycle"
                    block["cycle_period"] = step - previous
                    self._event({"type": "deterministic_cycle", "block_index": block["block_index"],
                                 "step": step+1, "period": step-previous})
                    break
                history.append((step, signature))
                history = history[-cfg.cycle_window:]
            before = dict(block["state_counts"])
            verified_before = block["verified_u_tokens"]
            times_before = dict(block["component_wall_s"])
            attempts_before = block["model_forward_attempts"]
            failed_before = block["model_forward_failed"]
            self.last_diagnostics["step"] = step + 1
            old_state = state.clone()
            old_confidence = confidence.clone() if cfg.temperature == 0 else None
            orig, shad, calls, fallback = self._forward(x, old_state, confidence)
            stats["model_forward_passes"] += calls
            stats["attention_fallback_steps"] += int(fallback)
            stats["steps"] += 1
            decoded = torch.where(old_state != M)[0]
            events: list[tuple[int, int]] = []
            promoted_count = revoked_count = revoked_c_count = demoted_count = 0
            verified_probability = None
            self.last_diagnostics["phase"] = "verification"
            with self._measure("verification", x.device):
                if decoded.numel():
                    probability = verify_probabilities(shad[decoded], x[0, p+decoded])
                    verified_probability = probability
                    updated = torch.full_like(decoded, M)
                    updated[probability > cfg.tau_c] = C
                    updated[probability > cfg.tau_u] = U
                    state[decoded] = updated
                    confidence[decoded] = probability
                    verified[decoded] = updated == U
                    revoked = decoded[updated == M]
                    x[0, p+revoked] = self.mask_id
                    confidence[revoked] = 0
                    was_c = old_state[decoded] == C
                    promoted_c = decoded[was_c & (updated == U)]
                    revoked_c = decoded[was_c & (updated == M)]
                    events = [(int(i), +1) for i in promoted_c.tolist()]
                    events += [(int(i), -1) for i in revoked_c.tolist()]
                    promoted_count = int(promoted_c.numel())
                    revoked_count = int(revoked.numel())
                    revoked_c_count = int(revoked_c.numel())
                    stats["promoted_tokens"] += promoted_count
                    stats["revoked_tokens"] += revoked_count
                    stats["u_reverified_tokens"] += int((old_state[decoded] == U).sum().item())
                    demoted_count = int(((old_state[decoded] == U) & (updated == C)).sum().item())
                    stats["demoted_tokens"] += demoted_count
            # Newly revoked positions wait for fresh logits on the next step.
            masked = torch.where(old_state == M)[0]
            count = new_u = 0
            eligible_count = None
            weights = sampled_probability = targets = None
            self.last_diagnostics["phase"] = "mixing_and_resampling"
            with self._measure("mixing_and_resampling", x.device):
                if masked.numel():
                    weights = adaptive_weights(size, events, prior=cfg.mix_prior, decay=cfg.mix_decay, device=device)
                    weight_sums.append(float(weights.mean().item()))
                    mixed = _mix_masked_logits(orig, shad, weights, masked)
                    proposed, probability = filtered_sample(mixed, cfg.temperature, cfg.top_p,
                                                            forbidden_token_ids=(self.mask_id,))
                    sampled_probability = probability
                    eligible = torch.where(probability > cfg.tau_c)[0]
                    eligible_count = int(eligible.numel())
                    if eligible.numel():
                        count = (min(cfg.transfer_limit, eligible.numel())
                                 if cfg.transfer_limit else eligible.numel())
                        selected = (eligible if count == eligible.numel() else
                                    eligible[torch.topk(probability[eligible], k=count).indices])
                        targets = masked[selected]
                        x[0, p+targets] = proposed[selected]
                        confidence[targets] = probability[selected]
                        state[targets] = torch.where(probability[selected] > cfg.tau_u, U, C)
                        verified[targets] = False
                        stats["candidate_proposals"] += count
                        new_u = int((state[targets] == U).sum().item())
                        stats["promoted_tokens"] += new_u
            # Reconstruct aggregate states from counts already needed by the
            # algorithm: summary mode never reduces an extra GPU tensor.
            revoked_u_count = revoked_count - revoked_c_count
            after = {"M": before["M"] + revoked_count - count,
                     "C": before["C"] - promoted_count - revoked_c_count + demoted_count + count-new_u,
                     "U": before["U"] + promoted_count-revoked_u_count-demoted_count+new_u}
            verified_after = after["U"] - new_u
            progress = verified_after - verified_before
            unchanged = not (count or promoted_count or revoked_count or demoted_count)
            stall_run = stall_run + 1 if progress <= 0 else 0
            block.update(state_counts=after, verified_u_tokens=verified_after,
                         steps=step+1, last_completed_step=step+1)
            block["stalled_steps"] += int(progress <= 0)
            block["unchanged_state_steps"] += int(unchanged)
            block["negative_verified_progress_steps"] += int(progress < 0)
            block["zero_eligible_steps"] += int(eligible_count == 0)
            block["max_stall_run"] = max(block["max_stall_run"], stall_run)
            revisits = 0
            if visits is not None and targets is not None:
                for position in targets.tolist():
                    revisits += int(visits[position] > 0)
                    visits[position] += 1
                    block["max_position_visits"] = max(block["max_position_visits"], visits[position])
                block["revisited_tokens"] += revisits
            cutoff = self._terminal_cutoff(x[0, p:])
            if (cutoff is not None and bool((state[:cutoff] == U).all()) and
                    bool((old_state[:cutoff] == U).all())):
                x = x[:, :p+cutoff]
                stats["stop_reason"] = "eos"
            # A fresh U proposal must undergo a verification round before exit.
            elif bool((state == U).all()) and bool((old_state == U).all()):
                stats["stop_reason"] = "verified"
            if (stats["stop_reason"] == "step_budget" and unchanged
                    and old_confidence is not None and torch.equal(confidence, old_confidence)):
                # Greedy decoding with identical tokens/states/confidence builds
                # the same inputs, mask, logits and next state. Repeating this
                # deterministic fixed point until the cap performs no work.
                stats["stop_reason"] = "fixed_point"
            if cfg.diagnostics != "summary":
                self._event({"type": "step", "block_index": block["block_index"], "step": step+1,
                             "state_before": before, "state_after": after,
                             "verified_before": verified_before, "verified_after": verified_after,
                             "verified_progress": progress, "stalled": progress <= 0,
                             "unchanged_state": unchanged, "stall_run": stall_run,
                             "proposed": int(count), "eligible": eligible_count,
                             "sampled_slots": int(masked.numel()), "promoted": promoted_count + new_u,
                             "revoked": revoked_count, "demoted": demoted_count,
                             "transitions": {"M_to_C": count-new_u, "M_to_U": new_u,
                                             "C_to_U": promoted_count, "C_to_M": revoked_c_count,
                                             "U_to_C": demoted_count, "U_to_M": revoked_u_count},
                             "revisited_tokens": revisits, "thresholds": dict(block["thresholds"]),
                             "confidence": self._tensor_summary(confidence[state != M]),
                             "verification_confidence": self._tensor_summary(verified_probability)
                             if verified_probability is not None else None,
                             "sampled_confidence": self._tensor_summary(sampled_probability)
                             if sampled_probability is not None else None,
                             "mix_weights": self._tensor_summary(weights) if weights is not None else None,
                             "prefix_length": int(p), "plain_input_length": int(p+size),
                             "packed_input_length": int(p+4*size),
                             "packed_attention_mask_elements": int((p+4*size)**2),
                             "last_actual_packed_input_length": block.get("last_packed_input_length"),
                             "last_actual_packed_mask_elements": block.get("last_packed_attention_mask_elements"),
                             "attention_fallback": bool(fallback),
                             "forward_calls": int(calls),
                             "actual_forward_attempts": block["model_forward_attempts"]-attempts_before,
                             "failed_forward_attempts": block["model_forward_failed"]-failed_before,
                             "component_wall_s": {key: value-times_before.get(key, 0.0)
                                                  for key, value in block["component_wall_s"].items()},
                             "stop_reason": stats["stop_reason"] if stats["stop_reason"] != "step_budget"
                             else ("step_budget" if step+1 == cfg.steps_per_block else "continue")})
            if stats["stop_reason"] in ("eos", "verified", "fixed_point"):
                break
        stats["completion_trigger"] = stats["stop_reason"]
        if stats["stop_reason"] in ("step_budget", "fixed_point", "deterministic_cycle"):
            if cfg.completion_policy == "legacy_force":
                x = self._finish_unverified(x, state, verified, stats)
                stats["stop_reason"] = "forced" if stats["forced_tokens"] else "step_budget"
            elif cfg.completion_policy == "force_then_verify":
                x = self._finish_unverified(x, state, verified, stats,
                    candidate_confidence=confidence, trim_terminal=False)
                x = self._verify_completion(x, state, confidence, verified, stats, p)
            elif cfg.completion_policy == "resample_then_verify":
                x = self._resample_then_verify(x, state, confidence, verified, stats, p)
        stats.update(self._completion_status(x, state, verified, p))
        if cfg.completion_policy != "legacy_force":
            stats["incomplete"] = not stats["verification_certified"]
            stats["stop_reason"] = "incomplete" if stats["incomplete"] else "verified"
        cutoff = self._terminal_cutoff(x[0, p:])
        if cutoff is not None and not stats["incomplete"]:
            x = x[:, :p+cutoff]
            stats["stop_reason"] = "eos"
        stats["termination_cause"] = ("completion_rejected"
            if stats["incomplete"] and stats["completion_remasked_tokens"] else
            "unverified_after_completion_budget"
            if stats["incomplete"] and cfg.completion_policy == "force_then_verify" else
            "unverified_at_decode_budget" if stats["incomplete"] else
            "legacy_forced_"+stats["stop_reason"] if stats["forced_tokens"] and cfg.completion_policy == "legacy_force" else
            "certified_"+stats["stop_reason"] if stats["verification_certified"] else stats["stop_reason"])
        stats["mix_weight_mean"] = round(sum(weight_sums)/len(weight_sums), 4) if weight_sums else 1.0
        block.update(complete=True, stop_reason=stats["stop_reason"],
                     output_tokens=int(x.shape[1]-p), **{key: stats[key] for key in
                     ("candidate_proposals", "promoted_tokens", "revoked_tokens", "demoted_tokens",
                      "forced_tokens", "forced_forward_passes", "u_reverified_tokens", "mix_weight_mean")})
        block.update({key: stats[key] for key in ("completion_policy", "completion_verification_budget",
            "completion_verification_steps", "completion_verification_forward_passes",
            "completion_reverified_tokens", "completion_rejected_tokens", "completion_remasked_tokens",
            "completion_trigger", "termination_cause", "incomplete", "verification_certified",
            "completion_unverified_tokens", "unresolved_mask_tokens")})
        return x, stats

    def generate(self, prompt_ids, num_new_tokens: int):
        self.last_diagnostics = self._new_diagnostics()
        self._current_block_diagnostics = None
        self._current_block_stats = None
        self._forced_record_count = 0
        self._pending_forced_records = []
        counter_keys = ("model_forward_passes", "candidate_proposals", "promoted_tokens",
                        "revoked_tokens", "forced_tokens", "demoted_tokens", "u_reverified_tokens",
                        "forced_forward_passes", "steps", "attention_fallback_steps",
                        "completion_verification_steps", "completion_verification_forward_passes",
                        "completion_reverified_tokens", "completion_rejected_tokens", "completion_remasked_tokens",
                        "completion_unverified_tokens", "unresolved_mask_tokens")
        totals = {key: 0 for key in counter_keys}
        totals.update({"blocks": [], "stop_reason": "length",
                       "completion_policy": self.config.completion_policy,
                       "completion_verification_budget": self.config.completion_verification_steps,
                       "incomplete": False, "verification_certified": True,
                       "termination_cause": "length",
                       "attention_layout": "dream_original_shadow_keys_plus_shifted_queries",
                       "implementation_status": "paper_core_shifted_dream_adapter_v2",
                       "compact_queries": self.config.compact_queries,
                       "rectangular_attention": self.last_diagnostics["rectangular_attention"],
                       "logit_projection_policy": self.logit_projection_policy,
                       "diagnostics": self.last_diagnostics})
        try:
            self.last_diagnostics["phase"] = "input_validation"
            if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1:
                raise ValueError("DARDDecoder requires one nonempty prompt (batch=1)")
            if num_new_tokens < 1:
                raise ValueError("num_new_tokens must be positive")
            if bool((prompt_ids == self.mask_id).any()):
                raise ValueError("DARD prompt must not contain MASK tokens")
            x = prompt_ids
            self.last_diagnostics.update(prompt_tokens=int(prompt_ids.shape[1]),
                                         requested_tokens=int(num_new_tokens))
            for start in range(0, num_new_tokens, self.config.block_size):
                x, info = self._block(x, min(self.config.block_size, num_new_tokens-start))
                for key in counter_keys:
                    totals[key] += info[key]
                totals["blocks"].append(info)
                totals["verification_certified"] &= info["verification_certified"]
                totals["termination_cause"] = info["termination_cause"]
                if info["incomplete"]:
                    totals["incomplete"] = True
                    totals["stop_reason"] = "incomplete"
                    break
                if info["stop_reason"] == "eos":
                    totals["stop_reason"] = "eos"
                    break
            self.last_diagnostics.update(complete=True, partial=False, phase="complete",
                                         stop_reason=totals["stop_reason"],
                                         output_tokens=int(x.shape[1]-prompt_ids.shape[1]))
            self.last_diagnostics.update({key: totals[key] for key in ("completion_policy",
                "completion_verification_budget", "completion_verification_steps",
                "completion_verification_forward_passes", "completion_reverified_tokens",
                "completion_rejected_tokens", "completion_remasked_tokens", "completion_unverified_tokens",
                "unresolved_mask_tokens", "incomplete", "verification_certified", "termination_cause")})
            return x[:, prompt_ids.shape[1]:], totals
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:2000],
                     "phase": self.last_diagnostics["phase"]}
            self.last_diagnostics.update(complete=False, partial=True, error=error,
                                         stop_reason="error", termination_cause="error",
                                         incomplete=True, verification_certified=False)
            if self._current_block_diagnostics is not None:
                block = self._current_block_diagnostics
                block.update(complete=False, stop_reason="error", termination_cause="error",
                             incomplete=True, verification_certified=False)
                # Existing counters may lag a failed forward; the independent
                # attempts counter includes that call and every failed probe.
                if self._current_block_stats is not None:
                    for key in counter_keys:
                        if key != "model_forward_passes":
                            block[key] = self._current_block_stats.get(key, 0)
            self._event({"type": "error", "block_index": len(self.last_diagnostics["blocks"])-1,
                         "step": self.last_diagnostics.get("step"), "error_type": error["type"],
                         "message": error["message"], "phase": error["phase"]})
            raise
        finally:
            try:
                self._serialize_forced_records()
            except Exception as exc:
                self.last_diagnostics["forced_record_error"] = {
                    "type": type(exc).__name__, "message": str(exc)[:2000]}
                if self.last_diagnostics["complete"]:
                    self.last_diagnostics.update(complete=False, partial=True, stop_reason="error")
                    raise
            self._current_block_diagnostics = None
            self._current_block_stats = None
