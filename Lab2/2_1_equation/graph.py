from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0.5, 2.2, 400)
plt.plot(x, np.exp2(x), label="$2^x$")
plt.plot(x, x ** 2 + 0.5, label="$x^2+0.5$")
plt.axvspan(1, 2, color="gray", alpha=0.15, label="интервал [1, 2]")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Графическое отделение корня")
plt.grid()
plt.legend()

path = Path(__file__).with_name("graph.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print("График сохранен:", path)
plt.show()
