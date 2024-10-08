from typing import List
from .backends import LoggerBackend


class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._backends: List[LoggerBackend] = []
        self._diabled_backends: List[LoggerBackend] = []
        self._initialized = True

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = Logger()
        return cls._instance

    def add_backend(self, backend: LoggerBackend):
        self._backends.append(backend)

    def log(self, log_dict, blacklist_types=None):
        if blacklist_types is None:
            blacklist_types = set()

        for backend in self._backends:
            if backend.__class__ in blacklist_types:
                continue

            backend.log(log_dict)

    def pop_backend(self, type=None):
        if type is None:
            return self._backends.pop()
        for i in range(len(self._backends)):
            index = -1 - i
            if isinstance(self._backends[index], type):
                return self._backends.pop(index)

    def disable(self, type):
        for backend in self._backends:
            if isinstance(backend, type):
                self._backends.remove(backend)
                self._diabled_backends.append(backend)

    def enable(self, type):
        for backend in self._diabled_backends:
            if isinstance(backend, type):
                self._diabled_backends.remove(backend)
                self._backends.append(backend)

    def close(self):
        for backend in self._backends:
            backend.close()
