"""Examples of using the cache_decorator module."""

from typing import Any
import logging
from time import sleep
from cache_decorator import Cache


@Cache(log_level="critical")
def cached_function_1(a: Any):
    """A function that sleeps for 2 seconds and returns a list with log level critical."""
    sleep(2)
    return [1, 2, 3]


@Cache(log_level="debug")
def cached_function_2(a: Any):
    """A function that sleeps for 2 seconds and returns a list with log level debug."""
    sleep(2)
    return [1, 2, 3]

@Cache(log_level="debug")
def cached_function_3(a: int) -> int:
    """A function that sleeps for 2 seconds and returns a list with log level debug."""
    sleep(2)
    return 1


if __name__ == "__main__":

    cached_function_1(1)
    cached_function_1(1)

    cached_function_1(2)

    logging.setLevel(logging.DEBUG)

    cached_function_2(1)
    cached_function_2(1)

    cached_function_2(2)
    
    t = cached_function_3(5)
