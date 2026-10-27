"""Quick checks of our model changes (run from the project root):
    python scripts/test_model.py
"""
import sys
from pathlib import Path

import torch

# Let Python find cli/submission/model.py (same trick train.py uses).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))
from submission.model import CourseEnhancer, CONFIG  # noqa: E402

torch.manual_seed(0)

m0 = CourseEnhancer(**CONFIG).eval()                       # lookahead 0
m8 = CourseEnhancer(**{**CONFIG, "lookahead": 8}).eval()   # lookahead 8

print("params, lookahead 0:", sum(p.numel() for p in m0.parameters()))   # want 3981581
print("params, lookahead 8:", sum(p.numel() for p in m8.parameters()))   # want 3981581
print("LSTM type with lookahead 8:", type(m8.model.enhance).__name__)     # want DelayedLSTM

with torch.no_grad():
    out = m8.enhance(torch.randn(1, 12345) * 0.1, 16000)
print("output shape:", tuple(out.shape))                                  # want (1, 12345)


def first_changed_frame(model, hop=100):
    """Change the input from frame 200 onward and return the earliest output
    frame that changes. A model that sees more future reacts earlier."""
    x = torch.randn(1, 16000 * 2) * 0.1          # 2 s = 320 frames
    x2 = x.clone()
    x2[:, 200 * hop:] += 0.5                      # change everything from frame 200 on
    with torch.no_grad():
        y0 = model(x)[1]
        y1 = model(x2)[1]
    changed = (y0 - y1).abs() > 0
    first_sample = changed.nonzero()[0, -1].item()
    return first_sample // hop

f0 = first_changed_frame(m0)
f8 = first_changed_frame(m8)
print("first changed output frame, lookahead 0:", f0)
print("first changed output frame, lookahead 8:", f8)
print("difference (want 8):", f0 - f8)

mb = CourseEnhancer(**{**CONFIG, "bidirectional": True, "rnn_units": 142}).eval()
print('params, BLSTM-142',sum(p.numel() for p in mb.parameters()))  # want 3981581
print('first changed output frame, BLSTM-142', first_changed_frame(mb))  # want 0

for la in (0, 8, -1):
    mc = CourseEnhancer(**{**CONFIG, "conformer": True, "lookahead": la}).eval()
    print(f"conformer lookahead {la}: params {sum(p.numel() for p in mc.parameters())}, "
          f"first changed frame {first_changed_frame(mc)}")