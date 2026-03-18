import math
import copy
import torch
import torch.nn.functional as F
from anticipation.config import *
from anticipation.vocab import *
from anticipation.sample import safe_logits, nucleus, future_logits
from anticipation import ops

# NOTE: This logic adapts the `generate_colab.ipynb` to support ANY general instrument mapped by `midi_program`.


def add_token_general(
    model,
    z,
    tokens,
    top_p,
    current_time,
    target_instrument_id: int,
    constrained=True,
    density_info=None,
    temperature=1.0,
    max_time_jump=500,
):
    assert len(tokens) % 3 == 0
    history = tokens.copy()
    lookback = max(len(tokens) - 1017, 0)
    history = history[lookback:]
    offset = ops.min_time(history, seconds=False)
    history[::3] = [tok - offset for tok in history[::3]]

    new_token = []
    with torch.no_grad():
        prefix = z + history
        inp = torch.tensor(prefix).unsqueeze(0)
        if torch.cuda.is_available():
            inp = inp.cuda()

        n_layer = model.config.n_layer if hasattr(model, "config") else 12
        logits, _, kv_cache = model(inp, past_key_values=[None] * n_layer)

        for i in range(3):
            if i == 0:
                logit = logits[0, -1].float()
            else:
                new_inp = torch.tensor([[new_token[-1]]])
                if torch.cuda.is_available():
                    new_inp = new_inp.cuda()
                logits, _, kv_cache = model(new_inp, past_key_values=kv_cache)
                logit = logits[0, -1].float()

            idx = len(prefix) + i - 1
            logit = safe_logits(logit, idx)

            if i == 0:
                logit = future_logits(logit, current_time - offset)
                if max_time_jump > 0:
                    max_allowed = (current_time - offset) + max_time_jump
                    mask = torch.zeros_like(logit, dtype=torch.bool)
                    for t in range(TIME_OFFSET, len(logit)):
                        if (t - TIME_OFFSET) > max_allowed:
                            mask[t] = True
                    valid = logit.clone()
                    valid[mask] = -float("inf")
                    if (valid > -float("inf")).any():
                        logit = valid
            elif i == 2 and constrained:
                # Constrain strictly to the specific target instrument ID instead of hardcoded violin.
                saved = logit[NOTE_OFFSET : NOTE_OFFSET + MAX_NOTE].clone()
                logit[NOTE_OFFSET : NOTE_OFFSET + MAX_NOTE] = -float("inf")

                # Broadest range constraint for an instrument. Low to High.
                # Using 0 to 127 to cover the whole instrument's range.
                vs = NOTE_OFFSET + target_instrument_id * 128 + 0
                ve = NOTE_OFFSET + target_instrument_id * 128 + 127 + 1

                # We need to make sure we don't go out of bounds if MAX_NOTE is smaller.
                max_bound = NOTE_OFFSET + MAX_NOTE
                if ve > max_bound:
                    ve = max_bound
                if vs < NOTE_OFFSET:
                    vs = NOTE_OFFSET

                if vs < ve:
                    logit[vs:ve] = saved[vs:ve]
                else:
                    # fallback if bound error
                    pass

                # Extend density logic here if needed (omitted for generic generalization unless needed)

            logit = logit / temperature
            logit = nucleus(logit, top_p)
            probs = F.softmax(logit, dim=-1)
            token = torch.multinomial(probs, 1)
            new_token.append(int(token))

    new_token[0] += offset
    return new_token


def generate_instrument_events(
    model,
    target_instrument_id: int,
    end_time: float,
    controls: list,
    top_p: float = 0.95,
):
    """
    Generates new events for the specific target instrument using the given controls limit.
    """
    end = int(TIME_RESOLUTION * end_time)
    delta = DELTA * TIME_RESOLUTION
    prompt = ops.pad([], 0)
    z = [ANTICIPATE]
    tokens, ctrls = ops.anticipate(prompt, ops.sort(controls))
    current_time = 0

    if ctrls:
        atime, adur, anote = ctrls[0:3]
        remaining = ctrls[3:]
        ant_time = atime - ATIME_OFFSET
    else:
        ant_time = math.inf
        remaining = []

    # Hard limiter for runaway generation
    max_iters = end * 2
    iter_count = 0

    while iter_count < max_iters:
        iter_count += 1

        while current_time >= ant_time - delta:
            tokens.extend([atime, adur, anote])
            if remaining:
                atime, adur, anote = remaining[0:3]
                remaining = remaining[3:]
                ant_time = atime - ATIME_OFFSET
            else:
                ant_time = math.inf

        new_token = add_token_general(
            model=model,
            z=z,
            tokens=tokens,
            top_p=top_p,
            current_time=max(0, current_time),
            target_instrument_id=target_instrument_id,
            constrained=True,
            density_info=None,
            max_time_jump=500,
        )
        new_time = new_token[0] - TIME_OFFSET
        if new_time >= end:
            break

        tokens.extend(new_token)
        dt = new_time - current_time
        assert dt >= 0
        current_time = new_time

    events, _ = ops.split(tokens)
    return ops.unpad(events)


def run_inference(
    model, target_instrument_id: int, source_midi_events: list, song_length: float
):
    """
    Wrapper for starting generation for a single track.
    source_midi_events controls what the model can reference while playing.
    """
    return generate_instrument_events(
        model, target_instrument_id, song_length, source_midi_events
    )
