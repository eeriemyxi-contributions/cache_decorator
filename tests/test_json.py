"""Submodule of the test suite for the JSON backend of the cache_decorator package."""

from time import sleep
import os
from typing import Any
from shutil import rmtree
from cache_decorator import Cache
from .utils import standard_test


@Cache(
    cache_path="{cache_dir}/{_hash}.json",
    cache_dir="./test_cache",
    dump_kwargs={"indent": 4},
    backup=False,
)
def cached_function(a: Any):
    """A function that sleeps for 2 seconds and returns a JSON-cached dictionary."""
    sleep(2)
    # WITH NON str keys the json library converts them to str so
    # the cache is not "transparent in this case"
    return {"a": 1, "b": [1, 2, 3]}


def test_json():
    """Test the JSON backend of the cache_decorator package."""
    result_1, result_2 = standard_test(cached_function)
    assert result_1 == result_2
    if os.path.exists("./test_cache"):
        rmtree("./test_cache")
