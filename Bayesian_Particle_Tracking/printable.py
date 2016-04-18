# Adapted from https://github.com/tdimiduk/yaml-serialize (updated for python 3)

from collections import OrderedDict
import inspect
import numpy as np

class Printable:
    @property
    def _dict(self):
        dump_dict = OrderedDict()

        for var in inspect.signature(self.__init__).parameters:
            if getattr(self, var, None) is not None:
                item = getattr(self, var)
                if isinstance(item, np.ndarray) and item.ndim == 1:
                    item = list(item)
                dump_dict[var] = item

        return dump_dict
    
    def __repr__(self):
        keywpairs = ["{0}={1}".format(k[0], repr(k[1])) for k in self._dict.items()]
        return "{0}({1})".format(self.__class__.__name__, ", ".join(keywpairs))

    def __str__(self):
        return self.__repr__()