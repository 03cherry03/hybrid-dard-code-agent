"""Offline backend contract tests; no model download or CUDA execution."""
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import torch

from dard_decoder import DARDConfig
import models


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0
    name_or_path = "fixture/tokenizer"
    init_kwargs = {"_commit_hash": "resolved-commit"}

    def __init__(self, wrapper="native"):
        self.wrapper = wrapper
        self.messages = []

    def get_chat_template(self):
        return self.wrapper

    def get_vocab(self):
        return {"source": 7, "more": 8, "<|im_end|>": 99}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        self.messages.append(messages)
        return self.wrapper + repr(messages)

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt" and add_special_tokens is False
        return {"input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long)}

    def decode(self, ids, *, skip_special_tokens):
        if skip_special_tokens:
            ids = [i for i in ids if i not in {0, 98, 99}]
        return "".join({7: "def f():\n", 8: "    return 1\n", 99: "<|im_end|>",
                        0: "<pad>", 98: "<|mask|>"}.get(x, "?") for x in ids)

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [7, 8] if text else []


class FakeGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=1024, mask_token_id=98,
                                      eos_token_id=99, _commit_hash="resolved-commit")
        self.config.to_dict = lambda: {"model_type": "fixture", "max_position_embeddings": 1024}
        self.generation_config = SimpleNamespace(eos_token_id=99, temperature=9.0,
                                                  repetition_penalty=2.0)
        self.options = None
        self.fail = False

    def forward(self, ids, *args, **kwargs):
        return SimpleNamespace(logits=torch.zeros(1, ids.shape[1], 100))

    def generate(self, *, input_ids, attention_mask, generation_config):
        self.options = generation_config
        self(input_ids)
        if self.fail:
            raise RuntimeError("fixture generation error")
        self(input_ids)
        self(input_ids)
        return SimpleNamespace(sequences=torch.cat((input_ids, torch.tensor([[7, 8, 99]])), dim=1))

    def diffusion_generate(self, input_ids, **kwargs):
        self.options = kwargs
        # Intentionally differs from the configured step budget: accounting
        # must observe calls rather than copying a configuration value.
        self(input_ids)
        self(input_ids)
        return SimpleNamespace(sequences=torch.cat((input_ids, torch.tensor([[7, 8, 99, 0, 0]])), dim=1))


@contextmanager
def fake_cuda():
    props = SimpleNamespace(total_memory=24 * 1024**3, uuid="GPU-fixture", pci_bus_id="0000:01:00.0")
    with patch.multiple(torch.cuda, device=lambda _: nullcontext(), is_available=lambda: True,
                        device_count=lambda: 2, get_device_name=lambda _: "Fixture GPU",
                        get_device_properties=lambda _: props, get_device_capability=lambda _: (8, 9),
                        synchronize=Mock(), reset_peak_memory_stats=Mock(),
                        max_memory_allocated=Mock(return_value=1024**3),
                        manual_seed=Mock(), empty_cache=Mock()) as mocks:
        yield mocks


def fixture_backend(kind, wrapper="native"):
    cls = models.ARBackend if kind == "ar" else models.DiffuCoderBackend
    backend = cls.__new__(cls)
    backend.torch = torch
    backend.device = torch.device("cpu")  # Deliberately bypass real CUDA constructor.
    backend.model = FakeGenerator()
    backend.tokenizer = FakeTokenizer(wrapper)
    backend.chat_template = wrapper
    backend.mask_id = None if kind == "ar" else 98
    backend.terminal_ids = [99]
    backend.pad_token_id = 0
    backend._closed = False
    backend._generation_config_class = SimpleNamespace
    backend._dard_decoder = None
    backend._dard_config = None
    backend._metadata = {"family": backend.family}
    return backend


def generate(backend, decoder, **kwargs):
    options = dict(seed=42, num_new_tokens=16, steps=73,
                   dard_config=DARDConfig(temperature=0, top_p=1), temperature=0, top_p=1)
    options.update(kwargs)
    return backend.generate("py", "def f(): ...", decoder, **options)


class TestModels(unittest.TestCase):
    def test_native_diffucoder_padding_policy_does_not_treat_chat_eos_as_padding(self):
        vocabulary = {"<|endoftext|>": 151643, "<|im_end|>": 151645, "<|dlm_pad|>": 151667}
        diffusion = models.terminal_token_policy("diffusion", vocabulary, 151643, [151643])
        self.assertEqual(diffusion["terminal_token_ids"], [151667])
        self.assertEqual(diffusion["termination_policy"], "diffucoder_native_dlm_pad_v1")
        self.assertEqual(diffusion["observed_end_marker_ids"], [151643, 151645, 151667])
        ar = models.terminal_token_policy("ar", vocabulary, 151643)
        self.assertEqual(ar["terminal_token_ids"], [151643, 151645, 151667])
        self.assertEqual(ar["termination_policy"], "configured_eos_and_chat_end_v1")

    def test_complete_token_trace_preserves_text_beyond_a_nonterminal_chat_marker(self):
        backend = fixture_backend("diffusion")
        class NativePadTokenizer(FakeTokenizer):
            def decode(self, ids, *, skip_special_tokens):
                if skip_special_tokens:
                    ids = [i for i in ids if i != 97]
                return "".join("<|dlm_pad|>" if i == 97 else super(NativePadTokenizer, self).decode(
                    [i], skip_special_tokens=skip_special_tokens) for i in ids)
        backend.tokenizer = NativePadTokenizer()
        backend.terminal_ids = [97]
        backend.termination_policy = "diffucoder_native_dlm_pad_v1"
        backend.observed_end_marker_ids = [97, 99]
        ids = [7, 99, 8, 97, 97]
        with fake_cuda(), patch.object(backend, "_generate_tokens", return_value=(torch.tensor(ids), {})):
            result = generate(backend, "standard")
        self.assertEqual(result["text"], "def f():\n    return 1")
        self.assertEqual(result["text_before_special_token_cleanup"],
                         "def f():\n<|im_end|>    return 1\n")
        self.assertEqual(result["generated_token_ids"], ids)
        self.assertEqual(result["raw_text_before_terminal"],
                         "def f():\n<|im_end|>    return 1\n<|dlm_pad|><|dlm_pad|>")
        self.assertEqual(result["raw_text_before_terminal_sha256"],
                         models._sha256(result["raw_text_before_terminal"]))
        self.assertEqual((result["terminal_token_id"], result["terminal_token_index"],
                          result["terminal_token_text"], result["tokens_after_terminal"]),
                         (97, 3, "<|dlm_pad|>", 1))
        markers = {row["token_id"]: row for row in result["end_marker_observations"]}
        self.assertEqual((markers[99]["first_index"], markers[99]["count"], markers[99]["terminates"]),
                         (1, 1, False))
        self.assertEqual((markers[97]["first_index"], markers[97]["count"], markers[97]["terminates"]),
                         (3, 2, True))

    def test_no_terminal_keeps_all_ids_and_records_no_fabricated_stop(self):
        backend = fixture_backend("ar")
        with fake_cuda(), patch.object(backend, "_generate_tokens", return_value=(torch.tensor([7, 8]), {})):
            result = generate(backend, "ar", num_new_tokens=2)
        self.assertEqual(result["generated_token_ids"], [7, 8])
        self.assertEqual(result["raw_text_before_terminal"], "def f():\n    return 1\n")
        self.assertIsNone(result["terminal_token_id"])
        self.assertIsNone(result["terminal_token_index"])
        self.assertEqual(result["end_marker_observations"], [])
        self.assertEqual(result["stop_reason"], "length")
        self.assertTrue(result["budget_exhausted"])

    def test_rendering_cleanup_does_not_hide_unfilled_mask_evidence(self):
        backend = fixture_backend("diffusion")
        with fake_cuda(), patch.object(backend, "_generate_tokens", return_value=(torch.tensor([7, 98, 8, 99]), {})):
            result = generate(backend, "standard")
        self.assertEqual(result["text"], "def f():\n    return 1")
        self.assertEqual(result["unfilled_mask_tokens"], 1)
        self.assertIn("<|mask|>", result["text_before_special_token_cleanup"])
        self.assertEqual(result["generated_token_ids"], [7, 98, 8, 99])

    def test_summary_diagnostics_do_not_add_forwards_or_per_forward_sync(self):
        backend = fixture_backend("ar")
        with fake_cuda(), patch.object(torch.cuda, "synchronize") as synchronize:
            result = generate(backend, "ar")
        self.assertEqual(result["model_forward_passes"], 3)
        self.assertEqual(synchronize.call_count, 3)  # same request boundaries as before instrumentation
        diag = result["diagnostics"]
        self.assertTrue(diag["complete"])
        self.assertFalse(result["instrumentation_affects_timing"])
        self.assertEqual(diag["forward_events"], [])
        self.assertEqual(diag["forward_shapes"]["sum_input_tokens"], 9)
        self.assertEqual(diag["forward_shapes"]["max_input_tokens"], 3)
        self.assertIsNone(diag["forward_profile_wall_s"])
        self.assertIn("output_decode", diag["stage_wall_s"])

    def test_profile_records_bounded_forward_timings_without_changing_output(self):
        backend = fixture_backend("ar")
        with fake_cuda():
            original = generate(backend, "ar")
            detailed = generate(backend, "ar", dard_config=DARDConfig(
                temperature=0, top_p=1, diagnostics="profile", max_step_records=1))
        self.assertEqual(original["text"], detailed["text"])
        self.assertEqual(original["model_forward_passes"], detailed["model_forward_passes"])
        self.assertTrue(detailed["instrumentation_affects_timing"])
        diag = detailed["diagnostics"]
        self.assertEqual(len(diag["forward_events"]), 1)
        self.assertEqual(diag["forward_records_dropped"], 2)
        self.assertGreaterEqual(diag["forward_profile_wall_s"], 0)
        self.assertIn("synchronized_wall_s", diag["forward_events"][0])
        self.assertFalse(backend.model._forward_pre_hooks)
        self.assertFalse(backend.model._forward_hooks)

    def test_failure_keeps_current_partial_counters_and_failure_phase(self):
        backend = fixture_backend("ar")
        with fake_cuda():
            generate(backend, "ar")
            backend.model.fail = True
            with self.assertRaisesRegex(RuntimeError, "fixture generation error") as caught:
                generate(backend, "ar")
        partial = caught.exception.generation_diagnostics
        self.assertEqual(partial["model_forward_passes"], 1)
        self.assertFalse(partial["diagnostics"]["complete"])
        self.assertEqual(partial["diagnostics"]["phase"], "model")
        self.assertEqual(partial["prompt_tokens"], 3)
        self.assertEqual(partial["gpu_peak_gib"], 1)
        self.assertEqual(partial["model_wall_s"], partial["diagnostics"]["stage_wall_s"]["model"])
        self.assertNotIn("text", partial)
        self.assertEqual(backend.last_generation_diagnostics, partial)
        self.assertFalse(backend.model._forward_pre_hooks)

    def test_keyboard_interrupt_keeps_partial_diagnostics_and_original_type(self):
        backend = fixture_backend("ar")
        with fake_cuda(), patch.object(backend.model, 'generate', side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt) as caught:
                generate(backend, "ar")
        partial = caught.exception.generation_diagnostics
        self.assertFalse(partial['measurement_complete'])
        self.assertEqual(partial['diagnostics']['error_type'], 'KeyboardInterrupt')
        self.assertEqual(partial['diagnostics']['phase'], 'model')
        self.assertEqual(backend.last_generation_diagnostics, partial)
        self.assertFalse(backend.model._forward_pre_hooks)

    def test_profile_precall_sync_failure_does_not_count_a_forward(self):
        backend = fixture_backend("ar")
        with fake_cuda(), patch.object(torch.cuda, "synchronize",
                side_effect=[None, None, RuntimeError("precall sync failed")]):
            with self.assertRaisesRegex(RuntimeError, "precall sync failed") as caught:
                generate(backend, "ar", dard_config=DARDConfig(diagnostics="profile"))
        partial = caught.exception.generation_diagnostics
        self.assertEqual(partial["model_forward_passes"], 0)
        self.assertEqual(partial["diagnostics"]["forward_events"], [])
        self.assertFalse(backend.model._forward_pre_hooks)
        self.assertFalse(backend.model._forward_hooks)

    def test_peak_query_failure_does_not_mask_generation_error_or_use_stale_peak(self):
        backend = fixture_backend("ar")
        backend.model.fail = True
        with fake_cuda(), patch.object(torch.cuda, "max_memory_allocated",
                side_effect=RuntimeError("CUDA context unavailable")):
            with self.assertRaisesRegex(RuntimeError, "fixture generation error") as caught:
                generate(backend, "ar")
        partial = caught.exception.generation_diagnostics
        self.assertNotIn("gpu_peak_gib", partial)
        self.assertEqual(partial["diagnostics"]["gpu_peak_unavailable"], "RuntimeError")
        with fake_cuda(), patch.object(torch.cuda, "reset_peak_memory_stats",
                side_effect=RuntimeError("reset failed")), patch.object(torch.cuda, "max_memory_allocated") as peak:
            with self.assertRaisesRegex(RuntimeError, "reset failed") as caught:
                generate(backend, "ar")
        peak.assert_not_called()
        self.assertNotIn("gpu_peak_gib", caught.exception.generation_diagnostics)

    def test_context_error_is_recorded_before_any_forward(self):
        backend = fixture_backend("ar")
        with fake_cuda(), self.assertRaises(ValueError) as caught:
            generate(backend, "ar", max_input_tokens=1)
        partial = caught.exception.generation_diagnostics
        self.assertEqual(partial["diagnostics"]["phase"], "context_budget_check")
        self.assertEqual(partial["model_forward_passes"], 0)
        self.assertEqual(partial["prompt_tokens"], 3)

    def test_repository_messages_are_preserved_and_have_shared_content_hash(self):
        messages = [{"role": "system", "content": "Return one repository action."},
                    {"role": "user", "content": "Fix module.py"},
                    {"role": "assistant", "content": '{"action":"read","path":"module.py"}'},
                    {"role": "user", "content": "Tool result: def f(): return 2"}]
        options = dict(seed=1, num_new_tokens=16, steps=4,
                       dard_config=DARDConfig(temperature=0, top_p=1), temperature=0, top_p=1)
        ar, diffusion = fixture_backend("ar", "ar"), fixture_backend("diffusion", "dream")
        with fake_cuda():
            first = ar.generate_messages(messages, "ar", **options)
            second = diffusion.generate_messages(messages, "standard", **options)
        self.assertEqual(ar.tokenizer.messages[0], messages)
        self.assertIsNot(ar.tokenizer.messages[0], messages)
        self.assertEqual(first["messages_sha256"], second["messages_sha256"])
        self.assertEqual(first["prompt_version"], models.MESSAGES_PROMPT_VERSION)
        self.assertEqual(first["model_forward_passes"], 3)

    def test_message_input_budget_is_checked_before_forward(self):
        backend = fixture_backend("ar")
        with fake_cuda(), self.assertRaisesRegex(ValueError, "shared input budget"):
            generate(backend, "ar", max_input_tokens=2)
        self.assertIsNone(backend.model.options)
        self.assertFalse(backend.model._forward_pre_hooks)

    def test_expired_deadline_does_not_tokenize_or_start_model(self):
        backend = fixture_backend("ar")
        with fake_cuda(), patch.object(models.time, "monotonic", return_value=10), \
                self.assertRaises(models.GenerationDeadlineExceeded):
            generate(backend, "ar", deadline=10)
        self.assertEqual(backend.tokenizer.messages, [])
        self.assertIsNone(backend.model.options)

    def test_deadline_interrupts_next_forward_and_always_removes_hook(self):
        for decoder in ("ar", "standard", "dard"):
            with self.subTest(decoder=decoder):
                backend = fixture_backend("ar" if decoder == "ar" else "diffusion")
                clock = [10.]
                forward = backend.model.forward

                def slow_forward(*args, **kwargs):
                    clock[0] += 2
                    return forward(*args, **kwargs)

                backend.model.forward = slow_forward
                # A fake DARD uses the same nn.Module invocation as the real decoder.
                class DeadlineDARD:
                    def __init__(self, model, *args, **kwargs):
                        self.model = model

                    def generate(self, tokens, **kwargs):
                        self.model(tokens)
                        self.model(tokens)
                        raise AssertionError("The second forward must be interrupted")

                with fake_cuda(), patch.object(models.time, "monotonic", side_effect=lambda: clock[0]), \
                        patch.object(models, "DARDDecoder", DeadlineDARD), \
                        self.assertRaises(models.GenerationDeadlineExceeded):
                    generate(backend, decoder, max_time=1)
                self.assertFalse(backend.model._forward_pre_hooks)
                self.assertEqual(clock[0], 12.)

    def test_finished_response_after_deadline_is_not_accepted(self):
        backend = fixture_backend("ar")
        clock = [10.]
        original = backend.model.generate

        def late_result(**kwargs):
            result = original(**kwargs)
            clock[0] = 12.
            return result

        backend.model.generate = late_result
        with fake_cuda(), patch.object(models.time, "monotonic", side_effect=lambda: clock[0]), \
                self.assertRaises(models.GenerationDeadlineExceeded):
            generate(backend, "ar", max_time=1)
        self.assertFalse(backend.model._forward_pre_hooks)

    def test_torch_uuid_is_normalized_for_nvidia_process_matching(self):
        bare = "12345678-1234-5678-90ab-1234567890ab"
        self.assertEqual(models._nvidia_uuid(bare), "GPU-" + bare)
        self.assertEqual(models._nvidia_uuid("GPU-" + bare), "GPU-" + bare)
        self.assertIsNone(models._nvidia_uuid("unsupported UUID representation"))
        self.assertIsNone(models._nvidia_uuid(None))

    def test_shared_messages_and_native_wrappers(self):
        ar = fixture_backend("ar", "AR_NATIVE")
        diffusion = fixture_backend("diffusion", "DREAM_NATIVE")
        with fake_cuda():
            first = generate(ar, "ar", old_code="def f(): return 2", error="assertion failed")
            second = generate(diffusion, "standard", old_code="def f(): return 2", error="assertion failed")
        self.assertEqual(ar.tokenizer.messages, diffusion.tokenizer.messages)
        self.assertEqual(first["messages_sha256"], second["messages_sha256"])
        self.assertNotEqual(first["rendered_prompt_sha256"], second["rendered_prompt_sha256"])
        self.assertNotEqual(first["chat_template_sha256"], second["chat_template_sha256"])
        for code, name in models.LANGUAGE_NAMES.items():
            self.assertIn(name, models.build_messages(code, "task")[1]["content"])
        with self.assertRaises(ValueError):
            models.build_messages("unknown", "task")

    def test_ar_uses_fresh_greedy_config_and_kv_cache(self):
        backend = fixture_backend("ar")
        with fake_cuda():
            generate(backend, "ar")
        config = vars(backend.model.options)
        self.assertTrue(config["use_cache"])
        self.assertFalse(config["do_sample"])
        self.assertEqual(config["repetition_penalty"], 1.0)
        self.assertEqual(config["eos_token_id"], [99])
        self.assertNotIn("temperature", config)
        self.assertNotIn("top_p", config)
        self.assertEqual(config["max_new_tokens"], 16)
        self.assertIsNot(backend.model.options, backend.model.generation_config)

    def test_sampling_uses_shared_temperature_and_no_hidden_top_k(self):
        backend = fixture_backend("ar")
        with fake_cuda():
            generate(backend, "ar", temperature=.4, top_p=.9)
        self.assertTrue(backend.model.options.do_sample)
        self.assertEqual(backend.model.options.temperature, .4)
        self.assertEqual(backend.model.options.top_p, .9)
        self.assertEqual(backend.model.options.top_k, 0)

    def test_visible_tokens_are_not_fixed_slots_and_forward_calls_are_measured(self):
        backend = fixture_backend("diffusion")
        with fake_cuda(), patch.object(models.time, "perf_counter", side_effect=[10., 11., 15., 20.]):
            result = generate(backend, "standard")
        self.assertEqual(result["model_forward_passes"], 2)
        self.assertEqual(result["steps"], 73)
        self.assertEqual(result["generated_tokens"], 5)
        self.assertEqual(result["output_slots"], 5)
        self.assertEqual(result["visible_output_tokens"], 2)
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(result["text"], "def f():\n    return 1")
        self.assertEqual(result["stop_reason"], "eos")
        self.assertFalse(result["budget_exhausted"])
        self.assertEqual(result["wall_s"], 10.)
        self.assertEqual(result["model_wall_s"], 4.)
        self.assertEqual(result["gpu_peak_gib"], 1.)
        self.assertEqual(len(backend.model._forward_pre_hooks), 0)

    def test_forward_hook_removed_when_generation_fails(self):
        backend = fixture_backend("ar")
        backend.model.fail = True
        with fake_cuda(), self.assertRaisesRegex(RuntimeError, "fixture generation error"):
            generate(backend, "ar")
        self.assertEqual(len(backend.model._forward_pre_hooks), 0)

    def test_context_is_rejected_instead_of_silently_truncated(self):
        backend = fixture_backend("ar")
        backend.model.config.max_position_embeddings = 8
        with fake_cuda(), self.assertRaisesRegex(ValueError, "no truncation"):
            generate(backend, "ar")
        self.assertIsNone(backend.model.options)

    def test_close_releases_decoder_model_and_tokenizer(self):
        backend = fixture_backend("diffusion")
        ref = weakref.ref(backend.model)
        backend._dard_decoder = SimpleNamespace(model=backend.model)
        with fake_cuda():
            backend.close()
            backend.close()
        self.assertIsNone(ref())
        self.assertIsNone(backend.tokenizer)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            generate(backend, "standard")

    def test_dard_receives_eos_and_is_reused_without_hiding_actual_calls(self):
        backend = fixture_backend("diffusion")

        class FakeDARD:
            def __init__(self, model, mask_id, config, *, eos_token_ids):
                self.model = model
                self.eos_token_ids = eos_token_ids
                self.calls = 0

            def generate(self, tokens, *, num_new_tokens):
                self.calls += 1
                self.model(tokens)
                return torch.tensor([[7, 8, 99]]), {"model_forward_passes": 500, "forced_tokens": 0}

        with fake_cuda(), patch.object(models, "DARDDecoder", FakeDARD):
            first = generate(backend, "dard")
            decoder = backend._dard_decoder
            second = generate(backend, "dard")
        self.assertIs(decoder, backend._dard_decoder)
        self.assertEqual(decoder.calls, 2)
        self.assertEqual(decoder.eos_token_ids, (99,))
        self.assertEqual(first["model_forward_passes"], 1)
        self.assertEqual(second["decoder_reported_forward_passes"], 500)

    def test_loader_pins_device_and_tokenizer_revision_without_download(self):
        model = FakeGenerator()
        model.parameters = lambda: iter([SimpleNamespace(device=torch.device("cuda:1"))])
        model.num_parameters = lambda: 7610000000
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=Mock(return_value=model))
        fake_transformers.AutoModel = SimpleNamespace(from_pretrained=Mock(return_value=model))
        fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(return_value=FakeTokenizer()))
        fake_transformers.BitsAndBytesConfig = SimpleNamespace
        fake_transformers.GenerationConfig = SimpleNamespace
        with fake_cuda(), patch.dict("sys.modules", {"transformers": fake_transformers}):
            backend = models.create_backend("ar", model_name="fixture/model", device="cuda:1",
                                            revision="requested-tag", local_files_only=True)
            kwargs = fake_transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
            self.assertEqual(kwargs["device_map"], {"": "cuda:1"})
            self.assertEqual(kwargs["torch_dtype"], torch.bfloat16)
            self.assertTrue(kwargs["local_files_only"])
            self.assertFalse(kwargs["trust_remote_code"])
            self.assertEqual(kwargs["revision"], "requested-tag")
            tok_kwargs = fake_transformers.AutoTokenizer.from_pretrained.call_args.kwargs
            self.assertEqual(tok_kwargs["revision"], "resolved-commit")
            meta = backend.metadata()
            self.assertEqual(meta["model_name"], "fixture/model")
            self.assertEqual(meta["resolved_revision"], "resolved-commit")
            self.assertEqual(meta["gpu_uuid"], "GPU-fixture")
            self.assertEqual(meta["gpu_pci_bus_id"], "0000:01:00.0")
            self.assertEqual(meta["parameter_count"], 7610000000)
            meta["versions"]["torch"] = "mutated"
            self.assertNotEqual(backend.metadata()["versions"]["torch"], "mutated")
            backend.close()
            with self.assertRaisesRegex(ValueError, "explicit single CUDA"):
                models.ARBackend(device="cpu")
            with self.assertRaisesRegex(ValueError, "explicit single CUDA"):
                models.ARBackend(device="cuda")


if __name__ == "__main__":
    unittest.main()
