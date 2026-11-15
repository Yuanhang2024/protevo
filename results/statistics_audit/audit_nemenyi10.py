import numpy as np
from scipy.stats import studentized_range
se = np.sqrt(4 * 5 / (6 * 9))
pairs = {"random-model": 1.1667, "random-llm": 1.9444, "random-knowledge": 1.5556,
         "model-llm": 0.7778, "model-knowledge": 0.3889, "llm-knowledge": 0.3889}
pub = {"random-model": 0.220784, "random-llm": 0.007635, "random-knowledge": 0.05175,
       "model-llm": 0.576973, "model-knowledge": 0.919384, "llm-knowledge": 0.919384}
for k, d in pairs.items():
    qn = d / se  # nemenyi statistic
    for label, dev in (("(x sqrt2)", qn * np.sqrt(2)), ("(plain)", qn)):
        p = 2 * (1 - studentized_range.cdf(dev, 4, np.inf))
        p = min(p, 1.0)
        if label == "(x sqrt2)":
            print(f"{k:16s} qn={qn:.4f} sr_dev={dev:.4f} p={p:.6f}  published={pub[k]}  diff={abs(p-pub[k]):.6f}")
