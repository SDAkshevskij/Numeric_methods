import math
from pathlib import Path


def read_data(path):
    data = list(map(float, path.read_text(encoding="utf-8").split()))
    n = int(data[0])
    if len(data) != 2 * n + 1:
        raise ValueError("Неверный формат входного файла.")
    return data[1:n + 1], data[n + 1:]


def solve(matrix, right):
    n = len(right)

    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot][column]) < 1e-14:
            raise ValueError("Нормальная система вырождена.")
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        right[column], right[pivot] = right[pivot], right[column]

        for row in range(column + 1, n):
            factor = matrix[row][column] / matrix[column][column]
            for j in range(column, n):
                matrix[row][j] -= factor * matrix[column][j]
            right[row] -= factor * right[column]

    result = [0.0] * n
    for i in range(n - 1, -1, -1):
        result[i] = (right[i] - sum(matrix[i][j] * result[j] for j in range(i + 1, n))) / matrix[i][i]
    return result


def least_squares(x, y, degree):
    matrix = [
        [sum(value ** (i + j) for value in x) for j in range(degree + 1)]
        for i in range(degree + 1)
    ]
    right = [sum(yi * xi ** i for xi, yi in zip(x, y)) for i in range(degree + 1)]
    return solve(matrix, right)


def polynomial(coefficients, x):
    return sum(coefficient * x ** degree for degree, coefficient in enumerate(coefficients))


def polynomial_text(coefficients):
    parts = []

    for degree, coefficient in enumerate(coefficients):
        variable = "" if degree == 0 else "x" if degree == 1 else f"x^{degree}"
        term = f"{abs(coefficient):.8g}{variable}"
        if not parts:
            parts.append(("-" if coefficient < 0 else "") + term)
        else:
            parts.append((" - " if coefficient < 0 else " + ") + term)
    return "".join(parts)


def validate(x, y):
    if len(x) < 4 or len(x) != len(y):
        raise ValueError("Нужно не менее четырех пар (x, y).")
    if len(set(x)) < 4 or not all(map(math.isfinite, x + y)):
        raise ValueError("Нужно не менее четырех разных конечных x.")


def main():
    x, y = read_data(Path(__file__).with_name("input.txt"))
    validate(x, y)
    results = []

    print("Исходные данные:")
    print("  i          x_i          y_i")
    for i, (xi, yi) in enumerate(zip(x, y)):
        print(f"{i:3d}  {xi:12.8f}  {yi:12.8f}")

    for degree in range(1, 4):
        coefficients = least_squares(x, y, degree)
        mse = sum((polynomial(coefficients, xi) - yi) ** 2 for xi, yi in zip(x, y)) / len(x)
        results.append((coefficients, mse))
        print(f"\nСтепень {degree}")
        print("Коэффициенты:", " ".join(f"a{i}={value:.10f}" for i, value in enumerate(coefficients)))
        print(f"P{degree}(x) = {polynomial_text(coefficients)}")
        print(f"MSE{degree} = {mse:.10e}")

    best = min(range(3), key=lambda i: results[i][1]) + 1
    print(f"\nЛучшее приближение: P{best}(x), у него наименьшая MSE.")


if __name__ == "__main__":
    main()
