import argparse
import math
from pathlib import Path

import numpy as np


def f(x):
    return 2 ** x - x ** 2 - 0.5


def df(x):
    return math.log(2) * 2 ** x - 2 * x


def phi(x):
    return math.sqrt(2 ** x - 0.5)


def dphi(x):
    return math.log(2) * 2 ** x / (2 * phi(x))


def simple_iteration(eps, x0):
    x = x0
    q = dphi(2)
    table = []

    for k in range(1, 1000):
        next_x = phi(x)
        error = q / (1 - q) * abs(next_x - x)
        table.append((k, next_x, error, abs(f(next_x))))
        x = next_x

        if error <= eps:
            return x, q, table

    raise RuntimeError("Метод простой итерации не сошелся.")


def newton(eps, x0):
    x = x0
    table = []

    for k in range(1, 1000):
        if abs(df(x)) < 1e-14:
            raise RuntimeError("Производная в методе Ньютона равна нулю.")
        next_x = x - f(x) / df(x)
        error = abs(next_x - x)
        table.append((k, next_x, error, abs(f(next_x))))
        x = next_x

        if error <= eps:
            return x, table

    raise RuntimeError("Метод Ньютона не сошелся.")


def print_result(name, root, table):
    print(f"\n{name}")
    print(" k          x             оценка ошибки       |f(x)|")

    for k, x, error, residual in table:
        print(f"{k:2d}  {x:14.10f}  {error:16.8e}  {residual:12.4e}")

    print(f"Корень: {root:.10f}")
    print(f"Количество итераций: {len(table)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("x0", type=float, help="начальное приближение из интервала [1, 2]")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("input.txt"))
    args = parser.parse_args()

    data = args.input.read_text(encoding="utf-8").split()
    if len(data) != 1:
        raise ValueError("В файле должна быть указана одна точность.")
    eps = float(data[0])

    if not math.isfinite(eps) or not 0 < eps < 1:
        raise ValueError("Точность должна находиться между 0 и 1.")
    if not math.isfinite(args.x0) or not 1 <= args.x0 <= 2:
        raise ValueError("Начальное приближение должно находиться в [1, 2].")

    simple_root, q, simple_table = simple_iteration(eps, args.x0)
    newton_root, newton_table = newton(eps, args.x0)

    print("Уравнение: 2^x - x^2 - 0.5 = 0")
    print(f"Точность: {eps:g}")
    print("Интервал корня: [1, 2]")
    print("Начальное приближение:", args.x0)
    print(f"Коэффициент сжатия q: {q:.6f}")
    print_result("Метод простой итерации", simple_root, simple_table)
    print_result("Метод Ньютона", newton_root, newton_table)

    roots = np.array([simple_root, newton_root])
    residuals = np.abs(np.exp2(roots) - roots ** 2 - 0.5)
    print("\nПроверка NumPy:")
    print("Невязки методов:", residuals)
    print("Различие решений:", np.max(np.abs(roots - roots[1])))


if __name__ == "__main__":
    main()
