import math
from pathlib import Path


def read_data(path):
    data = path.read_text(encoding="utf-8").split()
    count, position = int(data[0]), 1
    sets = []

    for _ in range(count):
        n = int(data[position])
        position += 1
        sets.append(list(map(float, data[position:position + n])))
        position += n

    if position != len(data) - 1:
        raise ValueError("Неверный формат входного файла.")
    return sets, float(data[-1])


def divided_differences(x, y):
    table = [[0.0] * len(x) for _ in x]

    for i, value in enumerate(y):
        table[i][0] = value
    for order in range(1, len(x)):
        for i in range(len(x) - order):
            table[i][order] = (
                table[i + 1][order - 1] - table[i][order - 1]
            ) / (x[i + order] - x[i])
    return table


def lagrange(x, y, point):
    result = 0.0

    for i in range(len(x)):
        term = y[i]
        for j in range(len(x)):
            if i != j:
                term *= (point - x[j]) / (x[i] - x[j])
        result += term
    return result


def newton(x, coefficients, point):
    result = coefficients[-1]

    for i in range(len(coefficients) - 2, -1, -1):
        result = result * (point - x[i]) + coefficients[i]
    return result


def factor(value):
    sign = "-" if value >= 0 else "+"
    return f"(x {sign} {abs(value):.6g})"


def lagrange_polynomial(x, y):
    terms = []

    for i in range(len(x)):
        denominator = math.prod(x[i] - x[j] for j in range(len(x)) if i != j)
        product = "".join(factor(x[j]) for j in range(len(x)) if i != j)
        terms.append(f"{y[i] / denominator:.8g}{product}")
    return " + ".join(terms).replace("+ -", "- ")


def newton_polynomial(x, coefficients):
    terms = [f"{coefficients[0]:.8g}"]

    for i in range(1, len(coefficients)):
        product = "".join(factor(value) for value in x[:i])
        terms.append(f"{coefficients[i]:.8g}{product}")
    return " + ".join(terms).replace("+ -", "- ")


def validate(sets, point):
    if not sets or not math.isfinite(point):
        raise ValueError("Входные данные пусты или содержат нечисловые значения.")
    for x in sets:
        if len(x) < 2 or len(set(x)) != len(x) or not all(map(math.isfinite, x)):
            raise ValueError("В каждом наборе нужны разные конечные узлы.")


def print_set(number, x, point):
    y = list(map(math.sin, x))
    table = divided_differences(x, y)
    coefficients = table[0]
    lagrange_value = lagrange(x, y, point)
    newton_value = newton(x, coefficients, point)
    exact = math.sin(point)

    print(f"\nНабор {number}")
    print("  i          x_i          y_i")
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"{i:3d}  {xi:12.8f}  {yi:12.8f}")

    print("L(x) =", lagrange_polynomial(x, y))
    print("\nТаблица разделенных разностей:")
    print("  i " + "".join(f"{'порядок ' + str(j):>16}" for j in range(len(x))))
    for i, row in enumerate(table):
        values = "".join(f"{row[j]:16.8f}" for j in range(len(x) - i))
        print(f"{i:3d} {values}")

    print("N(x) =", newton_polynomial(x, coefficients))
    print(f"L(x*) = {lagrange_value:.12f}")
    print(f"N(x*) = {newton_value:.12f}")
    print(f"f(x*) = {exact:.12f}")
    print(f"|L(x*) - f(x*)| = {abs(lagrange_value - exact):.6e}")
    print(f"|N(x*) - f(x*)| = {abs(newton_value - exact):.6e}")
    print(f"|L(x*) - N(x*)| = {abs(lagrange_value - newton_value):.6e}")


def main():
    path = Path(__file__).with_name("input.txt")
    sets, point = read_data(path)
    validate(sets, point)

    print("f(x) = sin(x)")
    print(f"x* = {point:.12f}")
    for number, x in enumerate(sets, 1):
        print_set(number, x, point)


if __name__ == "__main__":
    main()
