import atexit
import warnings

from zope.component import getSiteManager
from zope.component.registry import Components


class TestComponents:
    """
    A helper for providing a sterile registry when testing
    with :mod:`zope.component`.

    Instantiation will install an empty registry that will be returned
    by :func:`zope.component.getSiteManager`.
    """
    __test__ = False

    instances = set()
    atexit_setup = False

    def __init__(self):
        self.registry = Components('Testing')
        self.old = getSiteManager.sethook(lambda: self.registry)
        self.instances.add(self)
        if not self.__class__.atexit_setup:
            atexit.register(self.atexit)
            self.__class__.atexit_setup = True

    def uninstall(self):
        """
        Remove the sterile registry and replace it with the one that
        was in place before this :class:`TestComponents` was
        instantiated.
        """
        getSiteManager.sethook(self.old)
        self.instances.remove(self)

    @classmethod
    def atexit(cls):
        if cls.instances:
            warnings.warn(
                'TestComponents instances not uninstalled by shutdown!'
                )
