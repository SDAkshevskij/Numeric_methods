import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lab3_1 import divided_differences, lagrange, newton, read_data


sets, point = read_data(Path(__file__).with_name("input.txt"))
figure, axes = plt.subplots(1, len(sets), figsize=(7 * len(sets), 5), squeeze=False)

for number, (axis, nodes) in enumerate(zip(axes[0], sets), 1):
    values = list(map(math.sin, nodes))
    coefficients = divided_differences(nodes, values)[0]
    left, right = min(nodes), max(nodes)
    x = np.linspace(left - 0.1, right + 0.1, 500)
    axis.plot(x, np.sin(x), label="f(x) = sin(x)")
    axis.plot(x, [lagrange(nodes, values, value) for value in x], "--", label="Лагранж")
    axis.plot(x, [newton(nodes, coefficients, value) for value in x], ":", label="Ньютон")
    axis.scatter(nodes, values, color="black", zorder=3, label="узлы")
    axis.scatter(point, math.sin(point), color="red", zorder=4, label="x*")
    axis.set_title(f"Набор {number}")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid()
    axis.legend()

path = Path(__file__).with_name("graph.png")
figure.tight_layout()
figure.savefig(path, dpi=150)
print("График сохранен:", path)
plt.show()
