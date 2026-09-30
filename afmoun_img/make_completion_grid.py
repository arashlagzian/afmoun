from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required to write completion grids") from exc

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_tied_imagegpt import TiedImageGPT, autocast_dtype, read_metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate fixed ImageNet-32 top-half completions.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metrics-json", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--autocast", choices=["bf16", "fp16", "none"], default="")
    p.add_argument("--num-images", type=int, default=16)
    p.add_argument("--prime-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--cell-scale", type=int, default=4)
    return p.parse_args()


def load_tokens(path: Path, *, images: int, seq_len: int = 1024) -> torch.Tensor:
    required = int(images) * int(seq_len)
    available = path.stat().st_size // 2
    if available < required:
        raise ValueError(f"{path} has {available:,} uint16 tokens, need {required:,}")
    # RGB554 ids are in [0, 16383], so signed int16 is safe and avoids uint16 gaps in older PyTorch.
    toks = torch.from_file(str(path), shared=False, size=required, dtype=torch.int16)
    return toks.view(int(images), int(seq_len)).long()


def build_model(config: dict, metadata: dict, device: torch.device) -> TiedImageGPT:
    color_vocab = int(config.get("color_vocab_size", metadata.get("color_vocab_size", 16_384)))
    sos_token = int(config.get("sos_token", metadata.get("sos_token", color_vocab)))
    model = TiedImageGPT(
        color_vocab_size=color_vocab,
        sos_token=sos_token,
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        head_dim=int(config["head_dim"]),
        mlp_hidden=int(config["mlp_hidden"]),
        block_size=int(config.get("block_size", metadata.get("sequence_length", 1024))),
    ).float()
    return model.to(device)


def load_checkpoint(path: Path, data_dir: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    config = ckpt.get("config")
    if config is None:
        cfg_path = path.parent.parent / "config.json"
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
    metadata = read_metadata(data_dir)
    model = build_model(config, metadata, device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model, config, metadata


def rgb554_to_image(tokens: torch.Tensor, *, size: int = 32, scale: int = 4) -> Image.Image:
    t = tokens.detach().cpu().long().view(size, size)
    r = ((t // 512) & 31).to(torch.uint8)
    g = ((t // 16) & 31).to(torch.uint8)
    b = (t & 15).to(torch.uint8)
    rgb = torch.stack((r * 8 + 4, g * 8 + 4, b * 16 + 8), dim=-1).numpy()
    img = Image.fromarray(rgb, mode="RGB")
    if scale != 1:
        img = img.resize((size * scale, size * scale), Image.Resampling.NEAREST)
    return img


@torch.no_grad()
def completion_nll(model: TiedImageGPT, images: torch.Tensor, *, prime_tokens: int, device: torch.device, autocast: str) -> dict:
    images = images.to(device)
    sos = torch.full((images.shape[0], 1), model.sos_token, dtype=images.dtype, device=device)
    x_ids = torch.cat([sos, images[:, :-1]], dim=1)
    enabled = autocast != "none" and device.type == "cuda"
    dtype = autocast_dtype(autocast)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
        x = model.wte(x_ids)
        for block in model.blocks:
            x = block(x)
        x = model.norm(x)
        logits = F.linear(x, model.wte.weight[: model.color_vocab_size])
        target_logits = logits[:, int(prime_tokens) :, :]
        targets = images[:, int(prime_tokens) :]
        loss = F.cross_entropy(target_logits.reshape(-1, model.color_vocab_size), targets.reshape(-1))
    loss_f = float(loss.detach().float().item())
    return {
        "completion_loss": loss_f,
        "completion_ppl": math.exp(min(20.0, loss_f)),
        "completion_bits_per_token": loss_f / math.log(2.0),
        "completion_bits_per_dim": loss_f / (3.0 * math.log(2.0)),
        "completion_tokens": int(targets.numel()),
    }


@torch.no_grad()
def sample_completions(
    model: TiedImageGPT,
    images: torch.Tensor,
    *,
    prime_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    device: torch.device,
    autocast: str,
) -> torch.Tensor:
    torch.manual_seed(int(seed))
    generated = images[:, : int(prime_tokens)].to(device).clone()
    enabled = autocast != "none" and device.type == "cuda"
    dtype = autocast_dtype(autocast)
    for pos in range(int(prime_tokens), model.block_size):
        padded = torch.zeros((generated.shape[0], model.block_size), dtype=torch.long, device=device)
        padded[:, :pos] = generated
        sos = torch.full((generated.shape[0], 1), model.sos_token, dtype=torch.long, device=device)
        x_ids = torch.cat([sos, padded[:, :-1]], dim=1)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            x = model.wte(x_ids)
            for block in model.blocks:
                x = block(x)
            x = model.norm(x)
            logits = F.linear(x[:, pos, :], model.wte.weight[: model.color_vocab_size])
        logits = logits.float() / max(float(temperature), 1e-6)
        if int(top_k) > 0 and int(top_k) < logits.shape[-1]:
            vals, idx = torch.topk(logits, int(top_k), dim=-1)
            probs = torch.softmax(vals, dim=-1)
            next_tok = idx.gather(1, torch.multinomial(probs, num_samples=1))
        else:
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_tok], dim=1)
    return generated.cpu()


def write_grid(original: torch.Tensor, completed: torch.Tensor, out_path: Path, *, prime_tokens: int, scale: int) -> None:
    n = int(original.shape[0])
    labels = ["prime", "ground truth", "completion"]
    cell = 32 * int(scale)
    label_h = 18
    pad = 6
    grid = Image.new("RGB", (3 * cell + 4 * pad, n * (cell + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(grid)
    for col, label in enumerate(labels):
        draw.text((pad + col * (cell + pad), 2), label, fill=(0, 0, 0))
    for i in range(n):
        y = pad + label_h + i * (cell + label_h + pad)
        gt = original[i].clone()
        prime = gt.clone()
        prime[int(prime_tokens) :] = 0
        imgs = [prime, gt, completed[i]]
        for col, toks in enumerate(imgs):
            img = rgb554_to_image(toks, scale=scale)
            grid.paste(img, (pad + col * (cell + pad), y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, config, metadata = load_checkpoint(Path(args.checkpoint), data_dir, device)
    autocast = args.autocast or config.get("autocast", "bf16")
    eval_path = data_dir / metadata.get("eval_path", "eval_tokens_uint16.bin")
    images = load_tokens(eval_path, images=int(args.num_images), seq_len=int(metadata.get("sequence_length", 1024)))
    completed = sample_completions(
        model,
        images,
        prime_tokens=args.prime_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        device=device,
        autocast=autocast,
    )
    stats = completion_nll(model, images, prime_tokens=args.prime_tokens, device=device, autocast=autocast)
    stats.update({
        "checkpoint": str(args.checkpoint),
        "num_images": int(args.num_images),
        "prime_tokens": int(args.prime_tokens),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "sample_seed": int(args.seed),
    })
    out_path = Path(args.output)
    write_grid(images, completed, out_path, prime_tokens=args.prime_tokens, scale=args.cell_scale)
    metrics_path = Path(args.metrics_json) if args.metrics_json else out_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    print(f"wrote grid: {out_path}", flush=True)
    print(f"wrote metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
