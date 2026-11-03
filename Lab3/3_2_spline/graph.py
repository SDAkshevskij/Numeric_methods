from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lab3_2 import cubic_spline, read_data, spline_value


x, y, point = read_data(Path(__file__).with_name("input.txt"))
coefficients = cubic_spline(x, y)

for i in range(len(x) - 1):
    grid = np.linspace(x[i], x[i + 1], 100)
    values = [spline_value(x, coefficients, value)[0] for value in grid]
    plt.plot(grid, values, label=f"S{i + 1}, [{x[i]:g}, {x[i + 1]:g}]")

point_y, _ = spline_value(x, coefficients, point)
plt.scatter(x, y, color="black", zorder=3, label="узлы")
plt.scatter(point, point_y, color="red", zorder=4, label="(x*, S(x*))")
plt.xlabel("x")
plt.ylabel("S(x)")
plt.title("Естественный кубический сплайн")
plt.grid()
plt.legend()

path = Path(__file__).with_name("graph.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print("График сохранен:", path)
plt.show()
