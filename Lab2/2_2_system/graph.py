from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


a = 2
x1 = np.linspace(-1.1, 3.2, 500)
angle = np.linspace(0, 2 * np.pi, 500)

plt.plot(x1, a ** 3 / (x1 ** 2 + a ** 2), label="первое уравнение")
plt.plot(a / 2 + a * np.cos(angle), a / 2 + a * np.sin(angle), label="второе уравнение")
plt.fill_between([2.9, 3.1], 0.5, 0.7, color="gray", alpha=0.3, label="область начальной точки")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Графическое определение начального приближения")
plt.axis("equal")
plt.grid()
plt.legend()

path = Path(__file__).with_name("graph.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print("График сохранен:", path)
plt.show()
