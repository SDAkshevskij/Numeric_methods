import math
from pathlib import Path


def read_data(path):
    data = list(map(float, path.read_text(encoding="utf-8").split()))
    n = int(data[0])
    if len(data) != 2 * n + 2:
        raise ValueError("Неверный формат входного файла.")
    return data[1:n + 1], data[n + 1:2 * n + 1], data[-1]


def cubic_spline(x, y):
    n = len(x)
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    c = [0.0] * n

    if n > 2:
        m = n - 2
        lower = [0.0] + [h[i] for i in range(1, m)]
        main = [2 * (h[i] + h[i + 1]) for i in range(m)]
        upper = [h[i + 1] for i in range(m - 1)] + [0.0]
        right = [
            3 * ((y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i])
            for i in range(m)
        ]

        for i in range(1, m):
            factor = lower[i] / main[i - 1]
            main[i] -= factor * upper[i - 1]
            right[i] -= factor * right[i - 1]

        c[n - 2] = right[-1] / main[-1]
        for i in range(m - 2, -1, -1):
            c[i + 1] = (right[i] - upper[i] * c[i + 2]) / main[i]

    coefficients = []
    for i in range(n - 1):
        a = y[i]
        b = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d = (c[i + 1] - c[i]) / (3 * h[i])
        coefficients.append((a, b, c[i], d))
    return coefficients


def spline_value(x, coefficients, point):
    if point == x[-1]:
        interval = len(x) - 2
    else:
        interval = next(i for i in range(len(x) - 1) if x[i] <= point < x[i + 1])
    dx = point - x[interval]
    a, b, c, d = coefficients[interval]
    return a + b * dx + c * dx ** 2 + d * dx ** 3, interval


def validate(x, y, point):
    if len(x) < 2 or len(x) != len(y):
        raise ValueError("Нужно не менее двух пар (x, y).")
    if not all(map(math.isfinite, x + y + [point])):
        raise ValueError("Все значения должны быть конечными.")
    if any(x[i] >= x[i + 1] for i in range(len(x) - 1)):
        raise ValueError("Узлы x должны строго возрастать.")
    if not x[0] <= point <= x[-1]:
        raise ValueError("Точка x* должна лежать внутри таблицы.")


def main():
    x, y, point = read_data(Path(__file__).with_name("input.txt"))
    validate(x, y, point)
    coefficients = cubic_spline(x, y)
    value, interval = spline_value(x, coefficients, point)

    print("Исходные данные:")
    print("  i          x_i          y_i")
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"{i:3d}  {xi:12.8f}  {yi:12.8f}")

    print("\nКоэффициенты S_i(x) = a_i + b_i t + c_i t^2 + d_i t^3, t = x - x_i:")
    print(" i       интервал             a             b             c             d")
    for i, (a, b, c, d) in enumerate(coefficients):
        print(f"{i + 1:2d}  [{x[i]:g}, {x[i + 1]:g}]  {a:12.8f}  {b:12.8f}  {c:12.8f}  {d:12.8f}")

    print(f"\nx* = {point:g}")
    print(f"Интервал: {interval + 1}, [{x[interval]:g}, {x[interval + 1]:g}]")
    print(f"S(x*) = {value:.12f}")
    print("\nПроверка прохождения через узлы:")
    print("  i          S(x_i)          y_i       |S(x_i)-y_i|")
    for i, (xi, yi) in enumerate(zip(x, y)):
        spline_y, _ = spline_value(x, coefficients, xi)
        print(f"{i:3d}  {spline_y:14.10f}  {yi:12.8f}  {abs(spline_y - yi):14.6e}")

    left_second = 2 * coefficients[0][2]
    a, b, c, d = coefficients[-1]
    right_second = 2 * c + 6 * d * (x[-1] - x[-2])
    print(f"\nS''({x[0]:g}) = {left_second:.6e}")
    print(f"S''({x[-1]:g}) = {right_second:.6e}")


if __name__ == "__main__":
    main()
