"""Last-frame token copying with learned pointer and gate (J=1)."""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F

from history_anchored_occ.c0_model import encode_classes
from history_anchored_occ.voxel_warp import token_candidate_mask, voxel_centers, warp_occupancy

RADIUS = 2
SHIFTS = [(dx, dy) for dx in range(-RADIUS, RADIUS + 1) for dy in range(-RADIUS, RADIUS + 1)]
RING = torch.tensor([max(abs(dx), abs(dy)) for dx, dy in SHIFTS])


def shifted(x, fill=0.0):
    """x: n,C,50,50 -> n,C,25,50,50; [..., k, i, j] = x[..., i+dx_k, j+dy_k] (fill outside)."""
    padded = F.pad(x, (RADIUS,) * 4, value=fill)
    return torch.stack(
        [
            padded[:, :, RADIUS + dx : RADIUS + dx + 50, RADIUS + dy : RADIUS + dy + 50]
            for dx, dy in SHIFTS
        ],
        2,
    )


class CopyHead(nn.Module):
    def __init__(self, width=64, seed=2026):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.query = nn.Linear(128, width)
            self.key = nn.Linear(128, width)
            self.meta = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1))
            self.gate_u = nn.Linear(128, 32)
            self.gate = nn.Sequential(nn.Linear(39, 64), nn.ReLU(), nn.Linear(64, 1))
        with torch.no_grad():
            self.gate[-1].bias.fill_(-4.0)  # start close to C0
        self.width = width

    def forward(
        self, u, logits, tokens, embeddings, eligible, ages, horizon, gate_off=False, labels=None
    ):
        device = u.device
        # Keep the trained gate's normalization over the original five-frame layout.
        eligible = eligible.clone()
        eligible[:-1] = False
        with torch.no_grad():
            logp = F.log_softmax(logits.flatten(1), 0)
            cand = shifted(tokens[:, None].double(), 0.0)[:, 0].round().long().reshape(125, 2500)
            ok = (shifted(eligible[:, None].double(), 0.0)[:, 0] > 0.5).reshape(125, 2500)
            support = torch.gather(logp, 0, cand)
            ring = F.one_hot(RING.to(device), 3).to(u.dtype)
            meta = torch.cat(
                (
                    (support / 10).clamp(-3, 0).reshape(5, 25, 2500, 1),
                    (ages.to(u.dtype)[:, None, None, None] / 5).expand(5, 25, 2500, 1),
                    ring[None, :, None, :].expand(5, 25, 2500, 3),
                    torch.full((5, 25, 2500, 1), horizon / 3, device=device, dtype=u.dtype),
                ),
                -1,
            )
            any_ok = ok.any(0)
            probs = logp.exp()
            top = probs.max(0).values
            entropy = -(probs * logp).sum(0) / math.log(512)
        q = self.query(u.flatten(1).T)
        k = self.key(embeddings.flatten(2).transpose(1, 2))
        k = shifted(k.transpose(1, 2).reshape(5, self.width, 50, 50))
        dot = (q.T.reshape(1, self.width, 1, 2500) * k.reshape(5, self.width, 25, 2500)).sum(
            1
        ) / math.sqrt(self.width)
        score = (dot + self.meta(meta).squeeze(-1)).reshape(125, 2500)
        score = score.masked_fill(~ok, float("-inf"))
        alpha = torch.where(
            any_ok[None],
            torch.softmax(torch.where(any_ok[None], score, torch.zeros_like(score)), 0),
            torch.zeros_like(score),
        )
        alpha = torch.where(ok, alpha, torch.zeros_like(alpha))
        copy = []
        for start in range(0, 512, 64):
            with torch.no_grad():
                match = (
                    cand[None] == torch.arange(start, start + 64, device=device)[:, None, None]
                ).to(u.dtype)
            copy.append((match * alpha[None]).sum(1))
        copy = torch.cat(copy)
        stats = torch.stack(
            (
                (alpha * support).sum(0) / 10,
                copy.detach().max(0).values,
                top,
                entropy,
                ok.to(u.dtype).mean(0),
                torch.full_like(top, horizon / 3),
                any_ok.to(u.dtype),
            ),
            -1,
        )
        gate_logit = self.gate(torch.cat((self.gate_u(u.flatten(1).T), stats), -1)).squeeze(-1)
        g = torch.sigmoid(gate_logit) * any_ok.to(u.dtype)
        if gate_off:
            g = torch.zeros_like(g)
        mixture = (1 - g)[None] * probs + g[None] * copy
        out = torch.log(mixture.clamp_min(1e-30)).reshape(512, 50, 50)
        info = dict(gate=g.detach(), any_ok=any_ok)
        if labels is not None:
            lab = labels.reshape(1, 2500)
            positive = (cand == lab) & ok
            has = positive.any(0).to(u.dtype)
            valid = any_ok.to(u.dtype)
            mass = (alpha * positive.to(u.dtype)).sum(0)
            with torch.no_grad():
                target = (mass > probs.gather(0, lab)[0]).to(u.dtype)
                top1 = positive.to(u.dtype).gather(0, alpha.argmax(0, keepdim=True))[0]
            info.update(
                pointer_sum=(-torch.log(mass.clamp_min(1e-12)) * has).sum(),
                pointer_count=has.sum().detach(),
                gate_sum=(
                    F.binary_cross_entropy_with_logits(gate_logit, target, reduction="none") * valid
                ).sum(),
                gate_count=valid.sum().detach(),
                top1_correct=(top1 * has).sum().detach(),
                gate_target_mean=(target * valid).sum().detach(),
            )
        return out, info


@contextmanager
def bypass_reader(reader):
    handle = reader.register_forward_hook(lambda module, args, kwargs, out: args[0], with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


class CopyRoute:
    """Attach the copy head during one frozen forecaster rollout."""

    def __init__(self, entry, head, device):
        self.entry = entry
        self.head = head
        self.centers = voxel_centers(device)

    @contextmanager
    def active(self, content, timestamps_us, gate_off=False, labels=None):
        """Use history aligned to the predicted pose in each of the six future steps."""
        predictor = self.entry.predictor
        reader = predictor.query_extension
        conv = predictor.native.transformer.conv_out
        vae = predictor.native.vae
        if "_future_query_step" in predictor.__dict__:
            raise RuntimeError("Route already installed")
        original = predictor._future_query_step
        records = []
        t = timestamps_us[0].double()

        def step_fn(work, poses, bank, step):
            geometry = predictor._active_prior.read_geometry(step)
            with torch.no_grad():
                source_from_query = geometry.source_from_query[0]
                warps = [
                    warp_occupancy(content[0, j], source_from_query[j], self.centers)
                    for j in range(5)
                ]
                quantized, ids, _ = encode_classes(vae, torch.stack([w[0] for w in warps])[None])
                eligible = token_candidate_mask(torch.stack([w[1] for w in warps]))
            ages = (step * 0.5 + (t[4] - t) * 1e-6).to(content.device)
            calls = []

            def hook(module, args, output):
                calls.append(1)
                if len(calls) != 1 or output.shape[1:] != (512, 50, 50):
                    raise AssertionError("One native head call per step")
                u, logits = args[0][-1], output[-1]
                new, info = self.head(
                    u,
                    logits.detach(),
                    ids[0],
                    quantized[0],
                    eligible,
                    ages,
                    step * 0.5,
                    gate_off=gate_off,
                    labels=None if labels is None else labels[step - 1],
                )
                records.append(
                    dict(
                        step=step,
                        pre=logits.detach().argmax(0),
                        post=new.detach().argmax(0),
                        **info,
                    )
                )
                return torch.cat((output[:-1], new[None]), 0)

            handle = conv.register_forward_hook(hook)
            try:
                return original(work, poses, bank, step)
            finally:
                handle.remove()

        with bypass_reader(reader):
            predictor._future_query_step = step_fn
            try:
                yield records
            finally:
                del predictor._future_query_step
