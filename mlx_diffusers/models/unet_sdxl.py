"""SDXL conditional UNet.

Separate from the SD1.5 UNet because the config sets ``use_linear_projection``
and ``addition_embed_type=text_time``. The SD1.5 transformer rejects linear
projection, so this file owns that path. ``transformer_layers_per_block`` and
head counts come from the config. ``layers/`` is not modified.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_diffusers.layers import Conv2d, phase2_to_parameter
from mlx_diffusers.layers.embeddings import TimestepEmbedding, Timesteps
from mlx_diffusers.layers.norm import GroupNorm
from mlx_diffusers.layers.resnet import ResnetBlock2D
from mlx_diffusers.layers.updownsampling import Downsample2D, Upsample2D
from mlx_diffusers.models.transformer_2d import CONTINUOUS_NORM_EPS, BasicTransformerBlock
from mlx_diffusers.models.unet_2d_blocks import DownBlock2D, UpBlock2D


def _as_list(value, count: int):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * count


def _segment(enabled: bool, module, *args):
    """Run one block, checkpointing it when training so the backward graph stays small."""
    if not enabled:
        return module(*args)

    def run(*inner):
        return module(*inner)

    output = mx.checkpoint(run)(*args)
    mx.eval(output)
    return output


class LinearTransformer2D(nn.Module):
    """Continuous transformer whose ``proj_in`` / ``proj_out`` are linears."""

    def __init__(
        self,
        num_attention_heads: int,
        in_channels: int,
        num_layers: int,
        cross_attention_dim: int,
        norm_num_groups: int,
        norm_eps: float,
    ):
        super().__init__()
        head_dim = in_channels // num_attention_heads
        inner = num_attention_heads * head_dim
        self.norm = GroupNorm(norm_num_groups, in_channels, eps=CONTINUOUS_NORM_EPS)
        self.proj_in = nn.Linear(in_channels, inner)
        self.transformer_blocks = [
            BasicTransformerBlock(
                inner,
                num_attention_heads,
                head_dim,
                cross_attention_dim=cross_attention_dim,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ]
        self.proj_out = nn.Linear(inner, in_channels)
        self.gradient_checkpointing = False

    def __call__(self, hidden_states: mx.array, encoder_hidden_states: mx.array | None = None) -> mx.array:
        residual = hidden_states
        batch, _, height, width = hidden_states.shape
        hidden_states = self.norm(hidden_states)
        inner = hidden_states.shape[1]
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1)).reshape(batch, height * width, inner)
        hidden_states = self.proj_in(hidden_states)
        for block in self.transformer_blocks:
            if self.gradient_checkpointing:
                def run(states, context, layer=block):
                    return layer(states, encoder_hidden_states=context)

                hidden_states = mx.checkpoint(run)(hidden_states, encoder_hidden_states)
            else:
                hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
            # Materialize each block so a 10-layer SDXL transformer does not
            # keep the whole graph live on a 16GB machine.
            mx.eval(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = mx.transpose(hidden_states.reshape(batch, height, width, inner), (0, 3, 1, 2))
        return hidden_states + residual


def _resnet(in_channels: int, out_channels: int, temb_channels: int, groups: int, eps: float, act_fn: str) -> ResnetBlock2D:
    return ResnetBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        eps=eps,
        groups=groups,
        non_linearity=act_fn,
    )


class CrossAttnDownBlockXL(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int,
        transformer_layers: int,
        num_attention_heads: int,
        cross_attention_dim: int,
        groups: int,
        eps: float,
        act_fn: str,
        add_downsample: bool,
        downsample_padding: int,
    ):
        super().__init__()
        resnets = []
        attentions = []
        for index in range(num_layers):
            block_in = in_channels if index == 0 else out_channels
            resnets.append(_resnet(block_in, out_channels, temb_channels, groups, eps, act_fn))
            attentions.append(
                LinearTransformer2D(
                    num_attention_heads,
                    out_channels,
                    transformer_layers,
                    cross_attention_dim,
                    groups,
                    eps,
                )
            )
        self.resnets = resnets
        self.attentions = attentions
        self.downsamplers = None
        if add_downsample:
            self.downsamplers = [
                Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op")
            ]
        self.gradient_checkpointing = False

    def __call__(self, hidden_states, temb, encoder_hidden_states):
        states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = _segment(self.gradient_checkpointing, resnet, hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            states.append(hidden_states)
        if self.downsamplers is not None:
            hidden_states = _segment(self.gradient_checkpointing, self.downsamplers[0], hidden_states)
            states.append(hidden_states)
        return hidden_states, states


class CrossAttnUpBlockXL(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int,
        transformer_layers: int,
        num_attention_heads: int,
        cross_attention_dim: int,
        groups: int,
        eps: float,
        act_fn: str,
        add_upsample: bool,
    ):
        super().__init__()
        resnets = []
        attentions = []
        for index in range(num_layers):
            skip = in_channels if index == num_layers - 1 else out_channels
            block_in = prev_output_channel if index == 0 else out_channels
            resnets.append(_resnet(block_in + skip, out_channels, temb_channels, groups, eps, act_fn))
            attentions.append(
                LinearTransformer2D(
                    num_attention_heads,
                    out_channels,
                    transformer_layers,
                    cross_attention_dim,
                    groups,
                    eps,
                )
            )
        self.resnets = resnets
        self.attentions = attentions
        self.upsamplers = [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)] if add_upsample else None
        self.gradient_checkpointing = False

    def __call__(self, hidden_states, res_hidden_states, temb, encoder_hidden_states):
        for resnet, attn in zip(self.resnets, self.attentions):
            skip = res_hidden_states.pop()
            hidden_states = mx.concatenate([hidden_states, skip], axis=1)
            hidden_states = _segment(self.gradient_checkpointing, resnet, hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
        if self.upsamplers is not None:
            hidden_states = _segment(self.gradient_checkpointing, self.upsamplers[0], hidden_states)
        return hidden_states


class UNetMidBlockXL(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        transformer_layers: int,
        num_attention_heads: int,
        cross_attention_dim: int,
        groups: int,
        eps: float,
        act_fn: str,
    ):
        super().__init__()
        self.resnets = [
            _resnet(in_channels, in_channels, temb_channels, groups, eps, act_fn),
            _resnet(in_channels, in_channels, temb_channels, groups, eps, act_fn),
        ]
        self.attentions = [
            LinearTransformer2D(
                num_attention_heads,
                in_channels,
                transformer_layers,
                cross_attention_dim,
                groups,
                eps,
            )
        ]
        self.gradient_checkpointing = False

    def __call__(self, hidden_states, temb, encoder_hidden_states):
        hidden_states = _segment(self.gradient_checkpointing, self.resnets[0], hidden_states, temb)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = _segment(self.gradient_checkpointing, self.resnets[1], hidden_states, temb)
        return hidden_states


class DownBlockXL(DownBlock2D):
    """SD1.5 down block with an optional per-resnet checkpoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_checkpointing = False

    def __call__(self, hidden_states, temb):
        if not self.gradient_checkpointing:
            return super().__call__(hidden_states, temb)
        states = []
        for resnet in self.resnets:
            hidden_states = _segment(True, resnet, hidden_states, temb)
            states.append(hidden_states)
        if self.downsamplers is not None:
            hidden_states = _segment(True, self.downsamplers[0], hidden_states)
            states.append(hidden_states)
        return hidden_states, states


class UpBlockXL(UpBlock2D):
    """SD1.5 up block with an optional per-resnet checkpoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_checkpointing = False

    def __call__(self, hidden_states, res_hidden_states, temb):
        if not self.gradient_checkpointing:
            return super().__call__(hidden_states, res_hidden_states, temb)
        for resnet in self.resnets:
            skip = res_hidden_states.pop()
            hidden_states = mx.concatenate([hidden_states, skip], axis=1)
            hidden_states = _segment(True, resnet, hidden_states, temb)
        if self.upsamplers is not None:
            hidden_states = _segment(True, self.upsamplers[0], hidden_states)
        return hidden_states


class UNet2DConditionXL(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        if config.get("addition_embed_type") != "text_time":
            raise ValueError("UNet2DConditionXL requires addition_embed_type text_time")
        block_out = [int(value) for value in config["block_out_channels"]]
        down_types = list(config["down_block_types"])
        up_types = list(config["up_block_types"])
        count = len(down_types)
        layers = _as_list(config["layers_per_block"], count)
        head_source = config.get("num_attention_heads") or config["attention_head_dim"]
        heads = [int(value) for value in _as_list(head_source, count)]
        cross = [int(value) for value in _as_list(config["cross_attention_dim"], count)]
        transformers = [int(value) for value in _as_list(config.get("transformer_layers_per_block", 1), count)]
        groups = int(config["norm_num_groups"])
        eps = float(config["norm_eps"])
        act_fn = config["act_fn"]
        padding = int(config["downsample_padding"])
        time_dim = block_out[0] * 4
        self.conv_in = Conv2d(int(config["in_channels"]), block_out[0], kernel_size=3, padding=1)
        self.time_proj = Timesteps(block_out[0], bool(config["flip_sin_to_cos"]), float(config["freq_shift"]))
        self.time_embedding = TimestepEmbedding(block_out[0], time_dim, act_fn=act_fn)
        self.add_time_proj = Timesteps(
            int(config["addition_time_embed_dim"]),
            bool(config["flip_sin_to_cos"]),
            float(config["freq_shift"]),
        )
        self.add_embedding = TimestepEmbedding(int(config["projection_class_embeddings_input_dim"]), time_dim, act_fn=act_fn)

        down_blocks = []
        in_channels = block_out[0]
        for index, kind in enumerate(down_types):
            out_channels = block_out[index]
            add_downsample = index != len(block_out) - 1
            if kind == "CrossAttnDownBlock2D":
                block = CrossAttnDownBlockXL(
                    in_channels,
                    out_channels,
                    time_dim,
                    layers[index],
                    transformers[index],
                    heads[index],
                    cross[index],
                    groups,
                    eps,
                    act_fn,
                    add_downsample,
                    padding,
                )
            elif kind == "DownBlock2D":
                block = DownBlockXL(
                    in_channels,
                    out_channels,
                    time_dim,
                    layers[index],
                    groups,
                    eps,
                    act_fn,
                    add_downsample,
                    padding,
                )
            else:
                raise ValueError(f"unsupported down block {kind}")
            down_blocks.append(block)
            in_channels = out_channels
        self.down_blocks = down_blocks
        self.mid_block = UNetMidBlockXL(
            block_out[-1],
            time_dim,
            transformers[-1],
            heads[-1],
            cross[-1],
            groups,
            eps,
            act_fn,
        )

        reversed_channels = list(reversed(block_out))
        reversed_layers = list(reversed(layers))
        reversed_heads = list(reversed(heads))
        reversed_cross = list(reversed(cross))
        reversed_transformers = list(reversed(transformers))
        up_blocks = []
        output_channel = reversed_channels[0]
        for index, kind in enumerate(up_types):
            prev_output = output_channel
            output_channel = reversed_channels[index]
            skip_channels = reversed_channels[min(index + 1, len(reversed_channels) - 1)]
            num_layers = reversed_layers[index] + 1
            add_upsample = index != len(reversed_channels) - 1
            if kind == "UpBlock2D":
                block = UpBlockXL(
                    skip_channels,
                    prev_output,
                    output_channel,
                    time_dim,
                    num_layers,
                    groups,
                    eps,
                    act_fn,
                    add_upsample,
                )
            elif kind == "CrossAttnUpBlock2D":
                block = CrossAttnUpBlockXL(
                    skip_channels,
                    prev_output,
                    output_channel,
                    time_dim,
                    num_layers,
                    reversed_transformers[index],
                    reversed_heads[index],
                    reversed_cross[index],
                    groups,
                    eps,
                    act_fn,
                    add_upsample,
                )
            else:
                raise ValueError(f"unsupported up block {kind}")
            up_blocks.append(block)
        self.up_blocks = up_blocks
        self.conv_norm_out = GroupNorm(groups, block_out[0], eps)
        self.conv_out = Conv2d(block_out[0], int(config["out_channels"]), kernel_size=3, padding=1)

    def load_converted(self, weights: dict[str, mx.array]) -> None:
        pairs = [(key, phase2_to_parameter(key, value)) for key, value in weights.items()]
        self.load_weights(pairs, strict=True)
        self.eval()

    def enable_gradient_checkpointing(self) -> None:
        """Train with ``segmented_loss`` so the backward graph is one block."""
        self.gradient_checkpointing = True

    def __call__(
        self,
        sample: mx.array,
        timestep,
        encoder_hidden_states: mx.array,
        added_cond_kwargs: dict[str, mx.array],
    ) -> mx.array:
        if not isinstance(timestep, mx.array):
            timestep = mx.array(timestep)
        if timestep.ndim == 0:
            timestep = mx.full((sample.shape[0],), timestep)
        elif timestep.shape[0] != sample.shape[0]:
            timestep = mx.broadcast_to(timestep, (sample.shape[0],))
        # Match the checkpoint dtype. fp32 inputs against an fp16 UNet would upcast
        # every activation and blow the backward graph.
        compute = self.conv_in.weight.dtype
        sample = sample.astype(compute)
        encoder_hidden_states = encoder_hidden_states.astype(compute)
        emb = self.time_embedding(self.time_proj(timestep)).astype(compute)
        text_embeds = added_cond_kwargs["text_embeds"].astype(compute)
        time_ids = added_cond_kwargs["time_ids"]
        time_embeds = self.add_time_proj(time_ids.reshape((-1,))).reshape((text_embeds.shape[0], -1))
        aug = self.add_embedding(mx.concatenate([text_embeds, time_embeds.astype(compute)], axis=-1))
        emb = emb + aug
        sample = self.conv_in(sample)
        residuals = [sample]
        for block in self.down_blocks:
            if isinstance(block, CrossAttnDownBlockXL):
                sample, states = block(sample, emb, encoder_hidden_states)
            else:
                sample, states = block(sample, emb)
            residuals.extend(states)
        sample = self.mid_block(sample, emb, encoder_hidden_states)
        for block in self.up_blocks:
            count = len(block.resnets)
            skips = residuals[-count:]
            del residuals[-count:]
            if isinstance(block, CrossAttnUpBlockXL):
                sample = block(sample, skips, emb, encoder_hidden_states)
            else:
                sample = block(sample, skips, emb)
        sample = nn.silu(self.conv_norm_out(sample))
        return self.conv_out(sample)

    def segmented_loss(self, latent, embedding, noise, alpha, timestep, text_embeds, time_ids):
        """Backward one block at a time.

        ``mx.checkpoint`` still builds the full backward graph on eval, so a
        768 SDXL step peaked near 21GB. Saving block inputs and differentiating
        each block on its own keeps the live graph to one resnet or one
        attention sublayer. The transformer is not one checkpoint.
        """
        noisy = mx.sqrt(alpha) * latent + mx.sqrt(1.0 - alpha) * noise
        prediction, tape, prefixes = self._forward_tape(noisy, timestep, embedding, text_embeds, time_ids)
        diff = prediction.astype(mx.float32) - noise.astype(mx.float32)
        loss = mx.mean(diff * diff)
        cotan = mx.stop_gradient((2.0 * diff / diff.size).astype(prediction.dtype))
        mx.eval(loss, cotan)
        grads = self._backward_tape(tape, prefixes, prediction, cotan)
        mx.eval(loss, grads)
        return loss, grads

    def _forward_tape(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        if not isinstance(timestep, mx.array):
            timestep = mx.array(timestep)
        if timestep.ndim == 0:
            timestep = mx.full((sample.shape[0],), timestep)
        elif timestep.shape[0] != sample.shape[0]:
            timestep = mx.broadcast_to(timestep, (sample.shape[0],))
        compute = self.conv_in.weight.dtype
        sample = _keep(sample.astype(compute))
        context = _keep(encoder_hidden_states.astype(compute))
        emb = self.time_embedding(self.time_proj(timestep)).astype(compute)
        text_embeds = text_embeds.astype(compute)
        time_embeds = self.add_time_proj(time_ids.reshape((-1,))).reshape((text_embeds.shape[0], -1))
        aug = self.add_embedding(mx.concatenate([text_embeds, time_embeds.astype(compute)], axis=-1))
        emb = _keep(emb + aug)
        tape = []
        hidden = self._record(tape, self.conv_in, sample)
        residuals = [hidden]
        for block in self.down_blocks:
            hidden, states = self._down_tape(tape, block, hidden, emb, context)
            residuals.extend(states)
        hidden = self._mid_tape(tape, hidden, emb, context)
        for block in self.up_blocks:
            count = len(block.resnets)
            skips = residuals[-count:]
            del residuals[-count:]
            hidden = self._up_tape(tape, block, hidden, skips, emb, context)
        hidden = self._record(tape, lambda h: nn.silu(self.conv_norm_out(h)), hidden)
        hidden = self._record(tape, self.conv_out, hidden)
        prefixes = {id(module): name for name, module in self.named_modules()}
        return hidden, tape, prefixes

    def _down_tape(self, tape, block, hidden, emb, context):
        states = []
        if isinstance(block, CrossAttnDownBlockXL):
            for resnet, attn in zip(block.resnets, block.attentions):
                hidden = self._record(tape, lambda h, layer=resnet: layer(h, emb), hidden, module=resnet)
                hidden = self._attention_tape(tape, attn, hidden, context)
                states.append(hidden)
        else:
            for resnet in block.resnets:
                hidden = self._record(tape, lambda h, layer=resnet: layer(h, emb), hidden, module=resnet)
                states.append(hidden)
        if block.downsamplers is not None:
            hidden = self._record(tape, block.downsamplers[0], hidden)
            states.append(hidden)
        return hidden, states

    def _mid_tape(self, tape, hidden, emb, context):
        block = self.mid_block
        hidden = self._record(tape, lambda h, layer=block.resnets[0]: layer(h, emb), hidden, module=block.resnets[0])
        hidden = self._attention_tape(tape, block.attentions[0], hidden, context)
        hidden = self._record(tape, lambda h, layer=block.resnets[1]: layer(h, emb), hidden, module=block.resnets[1])
        return hidden

    def _up_tape(self, tape, block, hidden, skips, emb, context):
        attentions = block.attentions if isinstance(block, CrossAttnUpBlockXL) else [None] * len(block.resnets)
        for resnet, attn in zip(block.resnets, attentions):
            skip = skips.pop()
            channels = hidden.shape[1]
            joined = _keep(mx.concatenate([hidden, skip], axis=1))
            tape.append(("concat", hidden, skip, joined, channels))
            hidden = self._record(tape, lambda h, layer=resnet: layer(h, emb), joined, module=resnet)
            if attn is not None:
                hidden = self._attention_tape(tape, attn, hidden, context)
        if block.upsamplers is not None:
            hidden = self._record(tape, block.upsamplers[0], hidden)
        return hidden

    def _attention_tape(self, tape, attn, hidden, context):
        """One VJP per self-attention, cross-attention, and feed-forward.

        ``mlx-mfa`` is not installed. Scores stay on
        ``mx.fast.scaled_dot_product_attention``. Cutting the tape here keeps
        a 10-layer block from building a single backward graph. Query tiling
        inside one ``value_and_grad`` would not drop the peak.
        """
        batch, _, height, width = hidden.shape

        def project_in(states, layer=attn):
            states = layer.norm(states)
            inner = states.shape[1]
            states = mx.transpose(states, (0, 2, 3, 1)).reshape(batch, height * width, inner)
            return layer.proj_in(states)

        tokens = self._record(tape, project_in, hidden)
        for block in attn.transformer_blocks:
            def self_attn(states, layer=block):
                return layer.attn1(layer.norm1(states)) + states

            def cross_attn(states, layer=block):
                return layer.attn2(layer.norm2(states), encoder_hidden_states=context) + states

            def feed_forward(states, layer=block):
                return layer.ff(layer.norm3(states)) + states

            tokens = self._record(tape, self_attn, tokens, module=block.attn1)
            tokens = self._record(tape, cross_attn, tokens, module=block.attn2)
            tokens = self._record(tape, feed_forward, tokens, module=block.ff)

        def project_out(states, residual, layer=attn):
            inner = states.shape[-1]
            states = layer.proj_out(states)
            states = mx.transpose(states.reshape(batch, height, width, inner), (0, 3, 1, 2))
            return states + residual

        return self._record(tape, project_out, tokens, hidden)

    def _record(self, tape, fn, *inputs, module=None):
        output = _keep(fn(*inputs))
        if module is None and isinstance(fn, nn.Module):
            module = fn
        tape.append(("call", fn, inputs, output, module))
        return output

    def _backward_tape(self, tape, prefixes, prediction, cotan):
        acc = {key: mx.zeros_like(value) for key, value in tree_flatten(self.trainable_parameters())}
        pending = {id(prediction): cotan}
        for entry in reversed(tape):
            if entry[0] == "concat":
                _, left, right, output, channels = entry
                grad = pending.pop(id(output))
                _add_pending(pending, left, grad[:, :channels])
                _add_pending(pending, right, grad[:, channels:])
                continue
            _, fn, inputs, output, module = entry
            grad = pending.pop(id(output))
            input_grads, param_grads = _vjp(fn, grad, inputs, module)
            for tensor, tensor_grad in zip(inputs, input_grads):
                _add_pending(pending, tensor, tensor_grad)
            if param_grads is None:
                continue
            prefix = prefixes[id(module)]
            for key, value in tree_flatten(param_grads):
                full = f"{prefix}.{key}" if prefix else key
                if full not in acc:
                    raise KeyError(full)
                acc[full] = acc[full] + value
        mx.eval(*acc.values())
        mx.clear_cache()
        return tree_unflatten(list(acc.items()))


def _keep(value: mx.array) -> mx.array:
    value = mx.stop_gradient(value)
    mx.eval(value)
    return value


def _add_pending(pending: dict, tensor: mx.array, grad: mx.array) -> None:
    grad = mx.stop_gradient(grad)
    mx.eval(grad)
    current = pending.get(id(tensor))
    pending[id(tensor)] = grad if current is None else current + grad


def _vjp(fn, cotan, inputs, module):
    cotan = mx.stop_gradient(cotan)

    def scalar(*args):
        return mx.sum(fn(*args) * cotan)

    trainable = module is not None and bool(tree_flatten(module.trainable_parameters()))
    if not trainable:
        if len(inputs) == 1:
            input_grads = (mx.grad(scalar)(*inputs),)
        else:
            input_grads = mx.grad(scalar, argnums=tuple(range(len(inputs))))(*inputs)
        mx.eval(*input_grads)
        return input_grads, None

    def inner(param_tree, *args):
        module.update(param_tree)
        return scalar(*args)

    _, grads = mx.value_and_grad(inner, argnums=tuple(range(1 + len(inputs))))(
        module.trainable_parameters(),
        *inputs,
    )
    param_grads = grads[0]
    input_grads = grads[1:]
    mx.eval(param_grads, *input_grads)
    return input_grads, param_grads
