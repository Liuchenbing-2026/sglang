"""CPU loader contract: online buckets need not contain the whole checkpoint."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.models.laguna import LagunaForCausalLM
from sglang.srt.models.step3p5 import Step3p5ForCausalLM
from sglang.srt.models.step3p5_mtp import Step3p5MTP
from sglang.srt.models.step3p7 import Step3p7ForConditionalGeneration
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TinyLaguna(LagunaForCausalLM):
    """Small CPU parameter tree retaining the production checkpoint loader."""

    def __init__(self):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(
            num_experts=2, mlp_layer_types=["sparse"], tie_word_embeddings=False
        )
        self.model = nn.Module()
        self.model.start_layer, self.model.end_layer = 0, 1
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.experts = nn.Module()
        self.model.layers = nn.ModuleList([layer])
        experts = layer.mlp.experts
        experts.w13_weight = nn.Parameter(torch.zeros(2, 2, 1), requires_grad=False)
        experts.w2_weight = nn.Parameter(torch.zeros(2, 1), requires_grad=False)

        def load(param, weight, name, *, shard_id, expert_id):
            destination = (
                param[expert_id]
                if shard_id == "w2"
                else param[expert_id, 0 if shard_id == "w1" else 1]
            )
            destination.copy_(weight)

        experts.w13_weight.weight_loader = load
        experts.w2_weight.weight_loader = load

    @staticmethod
    def buckets():
        return [
            [
                (
                    f"model.layers.0.mlp.experts.{i}.{proj}.weight",
                    torch.tensor([float(i + 1)]),
                )
                for proj in ("gate_proj", "up_proj", "down_proj")
            ]
            for i in range(2)
        ]


class TinyStep(Step3p5ForCausalLM):
    def __init__(self):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(moe_num_experts=0)
        self.num_fused_shared_experts = 0
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.beta = nn.Parameter(torch.zeros(1), requires_grad=False)

    @staticmethod
    def buckets():
        return [[("alpha", torch.tensor([1.0]))], [("beta", torch.tensor([2.0]))]]


class TinyStepMTP(Step3p5MTP):
    def __init__(self):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(num_hidden_layers=0, num_nextn_predict_layers=1)
        self.draft_model_idx = 0
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(1, 1)
        self.model.enorm = nn.Module()
        self.model.enorm.weight = nn.Parameter(torch.zeros(1), requires_grad=False)

    @staticmethod
    def buckets():
        return [
            [("model.embed_tokens.weight", torch.ones(1, 1))],
            [("model.layers.0.enorm.weight", torch.tensor([2.0]))],
        ]


class TinyStepVL(Step3p7ForConditionalGeneration):
    def __init__(self):
        nn.Module.__init__(self)
        self.language_model = TinyStep()

    @staticmethod
    def buckets():
        return [
            [("language_model." + name, value) for name, value in bucket]
            for bucket in TinyStep.buckets()
        ]


class TestBucketedModelWeightUpdates(unittest.TestCase):
    def test_initial_load_requires_complete_checkpoint(self):
        for cls in (TinyLaguna, TinyStep, TinyStepMTP, TinyStepVL):
            with self.subTest(model=cls.__name__):
                model = cls()
                for _ in range(2):
                    with self.assertRaises((RuntimeError, AssertionError)):
                        model.load_weights(model.buckets()[0])
                model.load_weights(sum(model.buckets(), []))

    def test_distributed_update_accepts_individual_buckets(self):
        self.check_update("distributed")

    def test_tensor_update_accepts_individual_buckets(self):
        self.check_update("tensor")

    def test_flattened_update_accepts_individual_buckets(self):
        self.check_update("flattened")

    def test_update_without_checkpoint_loader(self):
        self.check_update("distributed", initial_load=False)

    def check_update(self, mode, initial_load=True):
        for cls in (TinyLaguna, TinyStep, TinyStepMTP, TinyStepVL):
            with self.subTest(model=cls.__name__):
                model = cls()
                reference = cls()
                reference.load_weights(sum(reference.buckets(), []))
                expected = {
                    name: value.detach().clone()
                    for name, value in reference.named_parameters()
                }
                if initial_load:
                    model.load_weights(sum(model.buckets(), []))
                for param in model.parameters():
                    param.data.zero_()
                updater = WeightUpdater(
                    tp_rank=0,
                    device="cpu",
                    gpu_id=0,
                    model_config=None,
                    custom_weight_loaders={},
                    get_model=lambda: model,
                    update_model_fields=lambda *a, **k: None,
                    recapture_cuda_graph=lambda: None,
                    get_model_runner=lambda: None,
                )
                with get_context().override_server_args(weight_cache_mode="off"):
                    for bucket in model.buckets():
                        if mode == "distributed":
                            success, message = updater.load_weights_from_distributed(
                                bucket
                            )
                        elif mode == "tensor":
                            with get_parallel().override(tp_rank=0):
                                success, message = updater.update_weights_from_tensor(
                                    bucket
                                )
                        else:
                            flat = FlattenedTensorBucket(named_tensors=bucket)
                            success, message = updater.update_weights_from_tensor(
                                {
                                    "flattened_tensor": flat.flattened_tensor,
                                    "metadata": flat.metadata,
                                },
                                load_format="flattened_bucket",
                            )
                        self.assertTrue(success, message)
                for name, param in model.named_parameters():
                    torch.testing.assert_close(param, expected[name])
                # An online update must not disable later full-load validation.
                with self.assertRaises((RuntimeError, AssertionError)):
                    model.load_weights(model.buckets()[0])


if __name__ == "__main__":
    unittest.main()
