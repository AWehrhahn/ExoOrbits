from functools import wraps, lru_cache

import numpy as np
import astropy.units as u
from astropy.time import Time


class hasCache:
    """ Parent Class for Classes that use the cache property """

    def cache_clear(self):
        for func in dir(self):
            if func.startswith("__"):
                continue
            func = getattr(self, func)
            if hasattr(func, "cache_clear"):
                func.cache_clear()


def cache(function):
    """ Cache Property that uses the lru_cache for numpy arrays. Only works with 1 argument however """
    # Switch order of array and self, because only the first argument is used for the cache
    @lru_cache(maxsize=32)
    def cached_wrapper(hashable_array, self):
        if hashable_array is not None:
            array = np.array(hashable_array)
            return function(self, array)
        else:
            return function(self)

    @wraps(function)
    def wrapper(self, array=None):
        if isinstance(array, np.ndarray):
            if array.ndim > 0:
                return cached_wrapper(tuple(array), self)
            else:
                return cached_wrapper(float(array), self)
        else:
            return cached_wrapper(array, self)

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


def resets_cache(function):
    """ Property for functions that should reset the cache, because they change the result """

    @wraps(function)
    def wrapper(self, value):
        if self._orbit is not None:
            self._orbit.cache_clear()
        return function(self, value)

    return wrapper


def time_input(function):
    """ Checks that the input value is an astropy Time """

    @wraps(function)
    def wrapper(self, value):
        value = Time(value)
        return function(self, value)

    return wrapper
