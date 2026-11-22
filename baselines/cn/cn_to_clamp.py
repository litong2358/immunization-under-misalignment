import re

import torch

_CLAMP_FLAT = {
    0:  "feature_extractor.frozen_extractor.0",
    1:  "feature_extractor.frozen_extractor.1",
    4:  "feature_extractor.frozen_extractor.4.0",
    5:  "feature_extractor.frozen_extractor.4.1",
    6:  "feature_extractor.frozen_extractor.5.0",
    7:  "feature_extractor.frozen_extractor.5.1",
    8:  "feature_extractor.frozen_extractor.6.0",
    9:  "feature_extractor.frozen_extractor.6.1",
    10: "feature_extractor.feature_extractor.0.0",
    11: "feature_extractor.feature_extractor.0.1",
}

_INV = {v: k for k, v in _CLAMP_FLAT.items()}


def cn_to_clamp(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith("feature_extractor.classifier.") or k.startswith("classifier.0."):
            tail = k.split("classifier.0.", 1)[-1] if "classifier.0." in k \
                else k.split("classifier.", 1)[-1].split(".", 1)[-1]
            out["classifier." + tail] = v
            continue
        m = re.match(r"(feature_extractor\.(?:frozen_extractor|feature_extractor)\.[0-9]+(?:\.[0-9]+)?)\.(.*)", k)
        if not m:
            continue
        pref, tail = m.group(1), m.group(2)
        if pref in _INV:
            out[f"feature_extractor.{_INV[pref]}.{tail}"] = v
    return out


def load_theta_I(net, ckpt, fmt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    if fmt == "cn":
        clamp_sd = cn_to_clamp(sd)
        dtype = next(net.parameters()).dtype
        clamp_sd = {k: v.to(dtype) for k, v in clamp_sd.items()}
        missing, unexpected = net.load_state_dict(clamp_sd, strict=False)
        miss_fe = [k for k in missing if k.startswith("feature_extractor")]
        assert not miss_fe, f"CN remap missing feature keys: {miss_fe[:5]}"
        assert not unexpected, f"CN remap unexpected keys: {unexpected[:5]}"
    else:
        net.load_immunized_state(ckpt)
    return net
