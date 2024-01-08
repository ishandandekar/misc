import functools as ft
from typing import Callable

ComposableFunction = Callable[[float], float]


def compose(*functions: ComposableFunction) -> ComposableFunction:
    return ft.reduce(lambda f, g: lambda x: g(f(x)), functions)


def add_three(x: float) -> float:
    return x + 3


def multiply_by_two(x: float) -> float:
    return x * 2


def add_n(x: float, n: float) -> float:
    return x + n


def main():
    x = 12
    x = add_three(x)
    x = add_three(x)
    x = multiply_by_two(x)
    x = multiply_by_two(x)
    print(f"Result: {x}")
    x = 12
    my_func = compose(add_three, add_three, multiply_by_two, multiply_by_two)
    print(f"Result: {my_func(x)}")


if __name__ == "__main__":
    main()
