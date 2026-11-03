from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lab3_3 import least_squares, polynomial, read_data


x, y = read_data(Path(__file__).with_name("input.txt"))
grid = np.linspace(min(x) - 0.5, max(x) + 0.5, 500)

plt.scatter(x, y, color="black", zorder=3, label="исходные данные")
for degree in range(1, 4):
    coefficients = least_squares(x, y, degree)
    plt.plot(grid, [polynomial(coefficients, value) for value in grid], label=f"P{degree}(x)")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Метод наименьших квадратов")
plt.grid()
plt.legend()

path = Path(__file__).with_name("graph.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
print("График сохранен:", path)
plt.show()
