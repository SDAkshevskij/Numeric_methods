import math
from pathlib import Path


def f(x):
    return x / (2 * x + 5)


def read_data(path):
    data = list(map(float, path.read_text(encoding="utf-8").split()))
    if len(data) != 4:
        raise ValueError("В файле нужны a, b, h1 и h2.")
    return data


def grid(a, b, h):
    n = round((b - a) / h)
    if n <= 0 or not math.isclose(n * h, b - a, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Шаг должен делить отрезок на целое число частей.")
    return [a + i * h for i in range(n + 1)]


def left_rectangles(a, b, h):
    x = grid(a, b, h)
    return h * sum(f(value) for value in x[:-1])


def right_rectangles(a, b, h):
    x = grid(a, b, h)
    return h * sum(f(value) for value in x[1:])


def midpoint_rectangles(a, b, h):
    x = grid(a, b, h)
    return h * sum(f((x[i] + x[i + 1]) / 2) for i in range(len(x) - 1))


def trapezoid(a, b, h):
    x = grid(a, b, h)
    return h * ((f(a) + f(b)) / 2 + sum(f(value) for value in x[1:-1]))


def simpson(a, b, h):
    x = grid(a, b, h)
    n = len(x) - 1
    if n % 2:
        raise ValueError("Для метода Симпсона число отрезков должно быть четным.")
    return h / 3 * (
        f(a) + f(b)
        + 4 * sum(f(x[i]) for i in range(1, n, 2))
        + 2 * sum(f(x[i]) for i in range(2, n, 2))
    )


def validate(a, b, h1, h2):
    if not all(map(math.isfinite, [a, b, h1, h2])) or a >= b or h1 <= h2 or h2 <= 0:
        raise ValueError("Нужно a < b и h1 > h2 > 0.")
    if a <= -2.5 <= b:
        raise ValueError("На отрезке функция имеет разрыв.")
    ratio = h1 / h2
    if not math.isclose(ratio, round(ratio), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Отношение h1/h2 должно быть целым.")


def main():
    a, b, h1, h2 = read_data(Path(__file__).with_name("input.txt"))
    validate(a, b, h1, h2)
    methods = [
        ("Левые прямоугольники", left_rectangles, 1),
        ("Правые прямоугольники", right_rectangles, 1),
        ("Средние прямоугольники", midpoint_rectangles, 2),
        ("Трапеции", trapezoid, 2),
        ("Симпсон", simpson, 4),
    ]
    ratio = h1 / h2
    exact = (2 * (b - a) - 5 * math.log((2 * b + 5) / (2 * a + 5))) / 4
    results = []

    print("f(x) = x / (2x + 5)")
    print(f"Отрезок: [{a:g}, {b:g}], h1 = {h1:g}, h2 = {h2:g}")
    print("\nМетод                         I(h1)          I(h2)    p           RR       оценка")
    for name, method, order in methods:
        coarse = method(a, b, h1)
        fine = method(a, b, h2)
        correction = (fine - coarse) / (ratio ** order - 1)
        refined = fine + correction
        estimate = abs(correction)
        results.append((name, fine, refined))
        print(f"{name:26s} {coarse:14.10f} {fine:14.10f}  {order:3d} {refined:14.10f} {estimate:11.3e}")

    print(f"\nТочное значение: {exact:.12f}")
    print("Абсолютные погрешности:")
    print("Метод                         |I(h2)-I|       |RR-I|")
    for name, fine, refined in results:
        print(f"{name:26s} {abs(fine - exact):12.6e} {abs(refined - exact):12.6e}")
    best = min(results, key=lambda item: abs(item[2] - exact))
    print(f"Лучший метод: {best[0]}.")


if __name__ == "__main__":
    main()
