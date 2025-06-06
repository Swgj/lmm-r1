from typing import Optional

import deepspeed
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from openrlhf.utils.logging_utils import init_logger

from .ring_attn_utils import set_hacked_position_ids, clear_hacked_position_ids
from openrlhf.models.lmm_kits.utils import get_generation_cls, smart_load_config
from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor

logger = init_logger(__name__)


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    exclude_modules=None,
    lora_dropout=0,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    value_head_prefix="score",
    device_map=None,
    packing_samples=False,
    **kwargs,
) -> nn.Module:
    """Retrieve a transformer model with a sequence regression head on top.

    This function loads a pretrained transformer model and attaches a linear layer for sequence regression.

    Args:
        model_name_or_path (str): Path to the pretrained model.
        model_type (str): Type of the model, either "reward" or "critic".
        bf16 (bool, optional): Enable bfloat16 precision. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        target_modules (list, optional): List of target modules for LoRA. Defaults to None.
        exclude_modules (list, optional): List of modules to exclude from LoRA. Defaults to None.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        normalize_reward (bool, optional): Normalize reward values. Defaults to False.
        use_flash_attention_2 (bool, optional): Use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed configuration for model partitioning across multiple GPUs when ZeRO-3 is enabled. Defaults to None.
        init_value_head (bool, optional): Initialize the value head. Defaults to False.
        value_head_prefix (str, optional): Prefix for the value head. Defaults to "score".
        device_map (dict, optional): Map of devices for model loading. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.

    Returns:
        nn.Module: A pretrained transformer model with a sequence regression head.
    """
    assert (
        model_type == "critic" or model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."

    config = smart_load_config(model_name_or_path)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    # Prioritize using the value_head_prefix in the model configuration.
    value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)
    logger.info(f"set value_head_prefix to `{value_head_prefix}`")
    base_class = get_generation_cls(config)
    base_pretrained_class = base_class.__base__
    if model_type == "reward":
        cls_class = _get_reward_model(base_class, value_head_prefix, packing_samples)
    else:
        cls_class = _get_critic_model(base_class, value_head_prefix, packing_samples)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        if isinstance(target_modules, list) and len(target_modules) == 1:
            target_modules = target_modules[0]
        if isinstance(exclude_modules, list) and len(exclude_modules) == 1:
            exclude_modules = exclude_modules[0]
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if value_head_prefix in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        value_head = getattr(model, value_head_prefix)
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model


def _get_reward_model(base_llm_model, value_head_prefix="score", packing_samples=False):
    class RewardModel(base_llm_model):
        supports_gradient_checkpointing = True

        def __init__(self, config):
            super().__init__(config)

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            pad_sequence=False,
            packed_seq_lens=None,
            visual_inputs=None,
        ) -> torch.Tensor:
            if visual_inputs is None:
                visual_inputs = {}
            inputs_embeds = super().get_inputs_embeds(input_ids, **visual_inputs)
            batch, seqlen = input_ids.size()
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
            forward_attention_mask = attention_mask
            if self.packing_samples:
                packed_position_ids = super().get_position_ids(input_ids,attention_mask=attention_mask,packing=True, **visual_inputs)
                input_ids, hacked_position_ids, _, ring_attn_pad_len, indices, inputs_embeds, split_position_ids = unpad_and_slice_tensor(
                    input_ids, attention_mask, ring_attn_group, inputs_embeds, packed_position_ids
                )
                position_ids = super().offset_split_position_ids(split_position_ids, hacked_position_ids) # this is true position_ids
                #position_ids is directly hacked into flash_attn_forward to distinguish between different sequences
                set_hacked_position_ids(hacked_position_ids)
                forward_attention_mask = None
            else:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = super().get_position_ids(input_ids,attention_mask=attention_mask,packing=False, **visual_inputs)

            outputs = super().forward(
                inputs_embeds=inputs_embeds, attention_mask=forward_attention_mask, position_ids=position_ids,output_hidden_states=True, **visual_inputs
            )
            clear_hacked_position_ids()
            if "last_hidden_state" in outputs:
                last_hidden_states = outputs["last_hidden_state"]
            elif "hidden_states" in outputs:
                last_hidden_states = outputs["hidden_states"][-1]
            else:
                raise ValueError("outputs should contain either last_hidden_state or hidden_states")

            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

            if self.packing_samples:
                values = gather_and_pad_tensor(values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
            reward = values.gather(dim=1, index=eos_indices).squeeze(1)

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            return (reward, outputs) if return_output else reward

    return RewardModel


def _get_critic_model(base_llm_model, value_head_prefix="score", packing_samples=False):
    class CriticModel(base_llm_model):
        supports_gradient_checkpointing = True

        def __init__(self, config):
            super().__init__(config)

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            action_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            values_allgather=False,
            packed_seq_lens=None,
            visual_inputs={},
        ) -> torch.Tensor:
            inputs_embeds = super().get_inputs_embeds(input_ids, **visual_inputs)
            batch, seqlen = input_ids.size()
            forward_attention_mask = attention_mask
            if self.packing_samples:
                packed_position_ids = super().get_position_ids(input_ids,attention_mask=attention_mask,packing=True, **visual_inputs)
                input_ids, hacked_position_ids, _, ring_attn_pad_len, indices, inputs_embeds, split_position_ids = unpad_and_slice_tensor(
                    input_ids, attention_mask, ring_attn_group, inputs_embeds, packed_position_ids
                )
                position_ids = super().offset_split_position_ids(split_position_ids, hacked_position_ids) # this is true position_ids
                #position_ids is directly hacked into flash_attn_forward to distinguish between different sequences
                set_hacked_position_ids(hacked_position_ids)
                forward_attention_mask = None
            else:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = super().get_position_ids(input_ids,attention_mask=attention_mask,packing=False, **visual_inputs)

            outputs = super().forward(
                inputs_embeds=inputs_embeds, attention_mask=forward_attention_mask, position_ids=position_ids,output_hidden_states=True, **visual_inputs
            )
            clear_hacked_position_ids()

            if action_mask is None:
                assert return_output
                return outputs

            if "last_hidden_state" in outputs:
                last_hidden_states = outputs["last_hidden_state"]
            elif "hidden_states" in outputs:
                last_hidden_states = outputs["hidden_states"][-1]
            else:
                raise ValueError("outputs should contain either last_hidden_state or hidden_states")
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)  # (1, total_seqs)

            if self.packing_samples:
                values = gather_and_pad_tensor(values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

            values = values[:, :-1]
            # normalize reward
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            action_values = values[:, -action_mask.shape[1] :] * action_mask.float()

            if return_output:
                return (action_values, outputs)
            else:
                return action_values

    return CriticModel
