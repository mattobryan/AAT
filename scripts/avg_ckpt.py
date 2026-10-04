"""Average the weights of several saved epochs of one official-RAMP run (task H5).

    python scripts/avg_ckpt.py OUT.pth ep_60_0.pth ep_70_0.pth ep_80_0.pth

Uniform mean of every float tensor (weights and BN running stats); integer buffers
(num_batches_tracked) are taken from the last checkpoint. Output is a raw state_dict that
aat.models.load_weights accepts. No training or data is touched.
"""
import sys
import torch


def load(p):
    sd = torch.load(p, map_location="cpu")
    return sd.get("model", sd.get("state_dict", sd))


def main(out, *paths):
    sds = [load(p) for p in paths]
    assert all(s.keys() == sds[0].keys() for s in sds), "checkpoints have different keys"
    avg = {}
    for k, v in sds[-1].items():
        avg[k] = torch.stack([s[k].double() for s in sds]).mean(0).to(v.dtype) if v.is_floating_point() else v.clone()
    torch.save(avg, out)
    print(f"averaged {len(paths)} checkpoints -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
