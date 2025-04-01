import numpy as np

import errors
import util


class AlgoBase:
    codename: str = None
    test_matrix = {}

    # Class methods.

    @classmethod
    def get_test_matrix(cls):
        m = {}
        for c in reversed(cls.__mro__):
            if issubclass(c, AlgoBase):
                m |= c.test_matrix
        return m

    @classmethod
    def get_subclasses(cls):
        for sub in cls.__subclasses__():
            yield from sub.get_subclasses()
            if sub.codename is not None:
                yield sub

    @classmethod
    def find_subclass(cls, *codenames):
        for sub in cls.get_subclasses():
            if sub.codename in codenames:
                return sub

    @classmethod
    def new(cls, mixins=[], **kwargs):
        bases = *mixins, cls
        codenames = [x.codename for x in bases]
        type_name = "_".join([*codenames, "algo"])

        new_type = type(type_name, bases, {})
        # Prevent this transient type from being chosen by find_subclass().
        new_type.codename = None

        return new_type(**kwargs)

    # General instance methods.

    def __init__(
        self,
        *,
        key=None,
        verbose=False,
        **extras,
    ):
        if extras:
            raise errors.InvalidConfig(
                suffix=f"unsupported params for this algorithm: {extras}"
            )

        self.key = key
        self.verbose = verbose
        self.set_record(None)

    def rng(self):
        return util.Random(self.key)

    def set_record(self, record):
        self.record = record

    def set_chan_num(self, chan_num):
        self.chan_num = chan_num

    def update_db(self, db):
        pass

    # Debugging.

    def debug(self, prefix, arr=None, type_=None, *, force=False):
        if not (self.verbose or force):
            return
        print(prefix, end="")

        if arr is not None:
            print(":", self.format_array(arr, type_))
        else:
            print()

    def format_array(self, arr, type_):
        """Used for debug printing."""
        if type_ == "bin":
            arr = list(map(lambda x: np.binary_repr(x, width=0), arr))
        elif type_ == "bits":
            arr = "".join(map(str, arr))
        return arr
