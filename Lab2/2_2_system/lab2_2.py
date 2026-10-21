import argparse
import math
from pathlib import Path

import numpy as np


A = 2


def f(x1, x2):
    f1 = (x1 ** 2 + A ** 2) * x2 - A ** 3
    f2 = (x1 - A / 2) ** 2 + (x2 - A / 2) ** 2 - A ** 2
    return f1, f2


def phi(x1, x2):
    value = A ** 2 - (x2 - A / 2) ** 2
    if value < 0:
        raise RuntimeError("Итерация вышла из области определения корня.")
    next_x1 = A / 2 + math.sqrt(value)
    next_x2 = A ** 3 / (x1 ** 2 + A ** 2)
    return next_x1, next_x2


def simple_iteration(eps, x0):
    x1, x2 = x0
    q1 = 0.5 / math.sqrt(A ** 2 - 0.5 ** 2)
    q2 = 2 * A ** 3 * 2.9 / (2.9 ** 2 + A ** 2) ** 2
    q = max(q1, q2)
    table = []

    for k in range(1, 1000):
        next_x1, next_x2 = phi(x1, x2)
        distance = max(abs(next_x1 - x1), abs(next_x2 - x2))
        error = q / (1 - q) * distance
        residual = max(map(abs, f(next_x1, next_x2)))
        table.append((k, next_x1, next_x2, error, residual))
        x1, x2 = next_x1, next_x2

        if error <= eps:
            return (x1, x2), q, table

    raise RuntimeError("Метод простой итерации не сошелся.")


def newton(eps, x0):
    x1, x2 = x0
    table = []

    for k in range(1, 1000):
        f1, f2 = f(x1, x2)
        j11, j12 = 2 * x1 * x2, x1 ** 2 + A ** 2
        j21, j22 = 2 * (x1 - A / 2), 2 * (x2 - A / 2)
        determinant = j11 * j22 - j12 * j21
        if abs(determinant) < 1e-14:
            raise RuntimeError("Матрица Якоби вырождена.")
        dx1 = (f2 * j12 - f1 * j22) / determinant
        dx2 = (f1 * j21 - f2 * j11) / determinant
        next_x1, next_x2 = x1 + dx1, x2 + dx2
        error = max(abs(dx1), abs(dx2))
        residual = max(map(abs, f(next_x1, next_x2)))
        table.append((k, next_x1, next_x2, error, residual))
        x1, x2 = next_x1, next_x2

        if error <= eps:
            return (x1, x2), table

    raise RuntimeError("Метод Ньютона не сошелся.")


def print_result(name, solution, table):
    print(f"\n{name}")
    print(" k         x1            x2         оценка ошибки     невязка")

    for k, x1, x2, error, residual in table:
        print(f"{k:2d}  {x1:12.8f}  {x2:12.8f}  {error:14.6e}  {residual:10.3e}")

    print(f"Решение: x1 = {solution[0]:.10f}, x2 = {solution[1]:.10f}")
    print(f"Количество итераций: {len(table)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("x1", type=float, help="первая координата начального приближения")
    parser.add_argument("x2", type=float, help="вторая координата начального приближения")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("input.txt"))
    args = parser.parse_args()

    data = args.input.read_text(encoding="utf-8").split()
    if len(data) != 1:
        raise ValueError("В файле должна быть указана одна точность.")
    eps = float(data[0])

    if not math.isfinite(eps) or not 0 < eps < 1:
        raise ValueError("Точность должна находиться между 0 и 1.")
    if not 2.9 <= args.x1 <= 3.1 or not 0.5 <= args.x2 <= 0.7:
        raise ValueError("Начальная точка должна лежать в [2.9, 3.1] x [0.5, 0.7].")

    x0 = args.x1, args.x2
    simple_solution, q, simple_table = simple_iteration(eps, x0)
    newton_solution, newton_table = newton(eps, x0)

    print("Система варианта 1, a = 2")
    print(f"Точность: {eps:g}")
    print("Начальное приближение:", x0)
    print(f"Коэффициент сжатия q: {q:.6f}")
    print_result("Метод простой итерации", simple_solution, simple_table)
    print_result("Метод Ньютона", newton_solution, newton_table)

    solutions = np.array([simple_solution, newton_solution])
    x1, x2 = solutions[:, 0], solutions[:, 1]
    residuals = np.maximum(
        np.abs((x1 ** 2 + A ** 2) * x2 - A ** 3),
        np.abs((x1 - A / 2) ** 2 + (x2 - A / 2) ** 2 - A ** 2),
    )
    print("\nПроверка NumPy:")
    print("Невязки методов:", residuals)
    print("Различие решений:", np.max(np.abs(solutions[0] - solutions[1])))


if __name__ == "__main__":
    main()
