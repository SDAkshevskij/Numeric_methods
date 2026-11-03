import math
from pathlib import Path


def read_data(path):
    data = list(map(float, path.read_text(encoding="utf-8").split()))
    n = int(data[0])
    if len(data) != 2 * n + 2:
        raise ValueError("Неверный формат входного файла.")
    return data[1:n + 1], data[n + 1:2 * n + 1], data[-1]


def derivatives(x, y, point):
    position = next((i for i, value in enumerate(x) if value >= point), len(x) - 1)
    start = min(max(position - 1, 0), len(x) - 3)
    x0, x1, x2 = x[start:start + 3]
    y0, y1, y2 = y[start:start + 3]
    first_difference = (y1 - y0) / (x1 - x0)
    second_difference = (
        (y2 - y1) / (x2 - x1) - first_difference
    ) / (x2 - x0)
    first = first_difference + second_difference * (2 * point - x0 - x1)
    second = 2 * second_difference
    return first, second, list(zip(x[start:start + 3], y[start:start + 3]))


def validate(x, y, point):
    if len(x) < 3 or len(x) != len(y):
        raise ValueError("Нужно не менее трех пар (x, y).")
    if not all(map(math.isfinite, x + y + [point])):
        raise ValueError("Все значения должны быть конечными.")
    if any(x[i] >= x[i + 1] for i in range(len(x) - 1)):
        raise ValueError("Узлы x должны строго возрастать.")
    if not x[0] <= point <= x[-1]:
        raise ValueError("Точка должна лежать внутри таблицы.")


def main():
    x, y, point = read_data(Path(__file__).with_name("input.txt"))
    validate(x, y, point)
    first, second, nodes = derivatives(x, y, point)

    print("Исходная таблица:")
    print("  i          x_i          y_i")
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"{i:3d}  {xi:12.8f}  {yi:12.8f}")

    print(f"\nТочка вычисления: x* = {point:g}")
    print("Использованные узлы:")
    for xi, yi in nodes:
        print(f"x = {xi:g}, y = {yi:g}")
    print(f"f'(x*)  = {first:.12f}")
    print(f"f''(x*) = {second:.12f}")


if __name__ == "__main__":
    main()
