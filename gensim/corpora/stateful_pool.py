"""
Python's builtin `multiprocessing.Pool` is designed to use stateless processes
for applying a function defined at the module scope. This means that `Process`
subclasses with run methods that differ from the default cannot be used with
the `Pool`. So when such a `Processes` subclass would be useful (often the
case for complicated run logic), a custom Pool implementation must be used.

This module provides a `StatefulProcessingPool` with an accompanying
`StatefulProcessor` class that can be overridden to provide stateful `run`
implementations.
"""
import multiprocessing as mp
from multiprocessing.pool import MaybeEncodingError, mapstar


class StatefulProcessor(mp.Process):
    """Process designed for use in `StatefulProcessingPool`.

    The `target` function is ignored in favor of a `run` method that reads
    from an input queue and writes to an output queue. The actual processing
    of the input is delegated to the `process` method, which should be overridden
    by subclasses.
    """

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={},
                 state_kwargs=None):
        super(StatefulProcessor, self).__init__(group, target, name, args, kwargs)
        state_kwargs = {} if state_kwargs is None else dict(state_kwargs)
        self.__dict__.update(state_kwargs)

    def process(self, text):
        """Process a document text; subclasses should override."""
        raise NotImplementedError

    def run(self):
        inqueue, outqueue, initializer, initargs, maxtasks = self._args

        assert maxtasks is None or (type(maxtasks) == int and maxtasks > 0)
        put = outqueue.put
        get = inqueue.get
        if hasattr(inqueue, '_writer'):
            inqueue._writer.close()
            outqueue._reader.close()

        if initializer is not None:
            initializer(*initargs)

        completed = 0
        while maxtasks is None or (maxtasks and completed < maxtasks):
            try:
                task = get()
            except (EOFError, OSError):
                mp.util.debug('worker got EOFError or OSError -- exiting')
                break

            if task is None:
                mp.util.debug('worker got sentinel -- exiting')
                break

            job, i, func, args, kwds = task
            try:
                if func is mapstar:
                    args = [(self.process, arg_list) for _, arg_list in args]
                else:
                    func = self.process
                result = (True, func(*args, **kwds))
            except Exception as e:
                result = (False, e)

            try:
                put((job, i, result))
            except Exception as e:
                wrapped = MaybeEncodingError(e, result[1])
                mp.util.debug("Possible encoding error while sending result: %s" % (
                    wrapped))
                put((job, i, (False, wrapped)))

            task = job = result = func = args = kwds = None
            completed += 1
        mp.util.debug('worker exiting after %d tasks' % completed)


class _PatchedPool(mp.pool.Pool):
    """Patch the builtin `Pool` to take a `processor_class` and `state_kwargs`;

    This allows users to use custom subclasses of `StatefulProcessor` to implement
    custom stateful `run` logic. Instances of `processor_class` are used to populate
    the worker pool; each is initialized using the standard `Process` init procedure,
    followed by setting custom attributes from `state_kwargs`.
    """

    def Process(self, *args, **kwds):
        kwds['state_kwargs'] = self._state_kwargs
        return self._processor_class(*args, **kwds)

    def __init__(self, processes=None, initializer=None, initargs=(), maxtasksperchild=None,
                 state_kwargs=None, processor_class=StatefulProcessor):
        self._state_kwargs = state_kwargs
        self._processor_class = processor_class
        super(_PatchedPool, self).__init__(processes, initializer, initargs, maxtasksperchild)

    def _repopulate_pool(self):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited. This is also used
        to initially populate the pool on init.
        """
        for i in range(self._processes - len(self._pool)):
            w = self.Process(args=(
                self._inqueue, self._outqueue,
                self._initializer, self._initargs, self._maxtasksperchild))

            self._pool.append(w)
            w.name = w.name.replace('Process', 'PoolWorker')
            w.daemon = True
            w.start()
            mp.util.debug('added worker')


class StatefulProcessingPool(object):
    """Pool that uses stateful worker processes; designed to worked with subclasses of
    `StatefulProcessor`. Implements the same interface as `multiprocessing.Pool`.
    """

    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, state_kwargs=None, processor_class=StatefulProcessor):
        """
        See `multiprocessing.Pool` for arguments not discussed here.

        Args:
            state_kwargs (dict): dictionary of attributes to initialize worker processes
                with. These will be set as attributes of the process object.
            processor_class (class): subclass of `StatefulProcessor` to use for worker
                processes.
        """
        self._pool = _PatchedPool(
            processes, initializer, initargs, maxtasksperchild, state_kwargs,
            processor_class)

    def apply(self, args=(), kwds={}):
        """Apply TextProcessor.process(*args, **kwds)."""
        return self._pool.apply_async(None, args, kwds).get()

    def map(self, iterable, chunksize=None):
        """Apply `TextProcessor.process` to each element in `iterable`, collecting the results
        in a list that is returned.
        """
        return self._pool.map(None, iterable, chunksize)

    def imap(self, iterable, chunksize=1):
        """Equivalent of `map()` -- can be MUCH slower than `Pool.map()`."""
        return self._pool.imap(None, iterable, chunksize)

    def imap_unordered(self, iterable, chunksize=1):
        """Like `imap()` method but ordering of results is arbitrary."""
        return self._pool.imap_unordered(None, iterable, chunksize)

    def apply_async(self, args=(), kwds={}, callback=None):
        """Asynchronous version of `apply()` method."""
        return self._pool.apply_async(None, args, kwds, callback)

    def map_async(self, iterable, chunksize=None, callback=None):
        """Asynchronous version of `map()` method."""
        return self._pool.map_async(None, iterable, chunksize, callback)

    def close(self):
        self._pool.close()

    def terminate(self):
        self._pool.terminate()

    def join(self):
        self._pool.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pool.terminate()
