import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, "."); sys.path.insert(0, "src")
import torch, torch.nn as nn
import train

results = []

# 1. fully-device model passes
m = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4)).to("cuda")
try:
    train._assert_model_on_device(m, "cuda")
    results.append(("model on cuda -> pass", True, ""))
except Exception as e:
    results.append(("model on cuda -> pass", False, str(e)))

# 2. stray parameter is caught and named
m2 = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)).to("cuda")
with torch.no_grad():
    m2[1].weight.data = m2[1].weight.data.to("cpu")
try:
    train._assert_model_on_device(m2, "cuda")
    results.append(("stray param -> raise", False, "no exception raised"))
except RuntimeError as e:
    named = "1.weight" in str(e) and "cpu" in str(e)
    results.append(("stray param -> raise+named", named, str(e)))

# 3. stray buffer is caught
m3 = nn.Module(); m3.register_buffer("b", torch.zeros(2))
try:
    train._assert_model_on_device(m3, "cuda")
    results.append(("stray buffer -> raise", False, "no exception raised"))
except RuntimeError as e:
    results.append(("stray buffer -> raise+named", "b" in str(e), str(e)))

# 4. moved tensors pass
ok_inputs = {
    "labels": torch.zeros(2, dtype=torch.long, device="cuda"),
    "boundaries": torch.zeros(2, dtype=torch.long, device="cuda"),
}
try:
    train._assert_tensors_on_device(ok_inputs, "cuda")
    results.append(("inputs on cuda -> pass", True, ""))
except Exception as e:
    results.append(("inputs on cuda -> pass", False, str(e)))

# 5. a tensor that was never moved is caught and named
bad_inputs = dict(ok_inputs)
bad_inputs["boundaries"] = torch.zeros(2, dtype=torch.long)
try:
    train._assert_tensors_on_device(bad_inputs, "cuda")
    results.append(("unmoved tensor -> raise", False, "no exception raised"))
except RuntimeError as e:
    results.append(("unmoved tensor -> raise+named", "boundaries" in str(e), str(e)))

# 6. index-free target ("cuda") must match indexed tensors ("cuda:0")
try:
    train._assert_tensors_on_device(
        {"x": torch.zeros(1, device="cuda:0")}, "cuda"
    )
    results.append(("cuda vs cuda:0 -> pass", True, ""))
except Exception as e:
    results.append(("cuda vs cuda:0 -> pass", False, str(e)))

print()
for name, ok, detail in results:
    print(("PASS  " if ok else "FAIL  ") + name)
    if detail:
        print("        " + detail[:160])
print()
print("ALL PASS" if all(r[1] for r in results) else "SOME FAILED")
