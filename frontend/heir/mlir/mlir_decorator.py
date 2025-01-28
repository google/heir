"""Decorator to define Numba "intrinsics" for MLIR functions"""

import uuid
import weakref
import collections
import functools

from numba.core import types, config
from numba.core.typing.templates import infer
from numba.core.serialize import ReduceMixin


class _MLIR(ReduceMixin):
    """
    Dummy callable for _MLIR
    """
    _memo = weakref.WeakValueDictionary()
    # hold refs to last N functions deserialized, retaining them in _memo
    # regardless of whether there is another reference
    _recent = collections.deque(maxlen=config.FUNCTION_CACHE_SIZE)

    __uuid = None

    def __init__(self, name, defn, prefer_literal=False, **kwargs):
        self._ctor_kwargs = kwargs
        self._name = name
        self._defn = defn
        self._prefer_literal = prefer_literal
        functools.update_wrapper(self, defn)

    @property
    def _uuid(self):
        """
        An instance-specific UUID, to avoid multiple deserializations of
        a given instance.

        Note this is lazily-generated, for performance reasons.
        """
        u = self.__uuid
        if u is None:
            u = str(uuid.uuid1())
            self._set_uuid(u)
        return u

    def _set_uuid(self, u):
        assert self.__uuid is None
        self.__uuid = u
        self._memo[u] = self
        self._recent.append(self)

    def _register(self):
        # _ctor_kwargs
        from numba.core.typing.templates import (make_intrinsic_template,
                                                 infer_global)

        template = make_intrinsic_template(self, self._defn, self._name,
                                           prefer_literal=self._prefer_literal,
                                           kwargs=self._ctor_kwargs)
        infer(template)
        infer_global(self, types.Function(template))

    def __call__(self, *args, **kwargs):
        """
        Calls the Python Impl
        """
        _, impl =  self._defn(None, *args, **kwargs)
        return impl(*args, **kwargs)

    def __repr__(self):
        return "<intrinsic {0}>".format(self._name)

    def __deepcopy__(self, memo):
        # NOTE: Intrinsic are immutable and we don't need to copy.
        #       This is triggered from deepcopy of statements.
        return self

    def _reduce_states(self):
        """
        NOTE: part of ReduceMixin protocol
        """
        return dict(uuid=self._uuid, name=self._name, defn=self._defn)

    @classmethod
    def _rebuild(cls, uuid, name, defn):
        """
        NOTE: part of ReduceMixin protocol
        """
        try:
            return cls._memo[uuid]
        except KeyError:
            llc = cls(name=name, defn=defn)
            llc._register()
            llc._set_uuid(uuid)
            return llc


def mlir(*args, **kwargs):
    """
    TODO (#1162): update this doc
    """
    # Make inner function for the actual work
    def _mlir(func):
        name = getattr(func, '__name__', str(func))
        llc = _MLIR(name, func, **kwargs)
        llc._register()
        return llc

    if not kwargs:
        # No option is given
        return _mlir(*args)
    else:
        # options are given, create a new callable to recv the
        # definition function
        def wrapper(func):
            return _mlir(func)
        return wrapper
