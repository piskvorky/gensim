import multiprocessing as mp
from multiprocessing import util
from multiprocessing.pool import MaybeEncodingError, mapstar


class TextProcessor(mp.Process):
    """Generic process whose target function processes batches of texts."""

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={},
                 state_kwargs=None):
        super(TextProcessor, self).__init__(group, target, name, args, kwargs)
        state_kwargs = {} if state_kwargs is None else dict(state_kwargs)
        self.init_state(state_kwargs)

    def init_state(self, state_kwargs):
        for name, value in state_kwargs.items():
            setattr(self, name, value)

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
                util.debug('worker got EOFError or OSError -- exiting')
                break

            if task is None:
                util.debug('worker got sentinel -- exiting')
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
                util.debug("Possible encoding error while sending result: %s" % (
                    wrapped))
                put((job, i, (False, wrapped)))

            task = job = result = func = args = kwds = None
            completed += 1
        util.debug('worker exiting after %d tasks' % completed)


class _PatchedPool(mp.pool.Pool):

    def Process(self, *args, **kwds):
        kwds['state_kwargs'] = self._state_kwargs
        return self._processor_class(*args, **kwds)

    def __init__(self, processes=None, initializer=None, initargs=(), maxtasksperchild=None,
                 state_kwargs=None, processor_class=TextProcessor):
        self._state_kwargs = state_kwargs
        self._processor_class = processor_class
        super(_PatchedPool, self).__init__(processes, initializer, initargs, maxtasksperchild)

    def _repopulate_pool(self):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        for i in range(self._processes - len(self._pool)):
            w = self.Process(args=(self._inqueue, self._outqueue,
                                   self._initializer,
                                   self._initargs, self._maxtasksperchild))
            self._pool.append(w)
            w.name = w.name.replace('Process', 'PoolWorker')
            w.daemon = True
            w.start()
            util.debug('added worker')


class TextProcessingPool(object):
    """Pool for processing batches of texts using TextProcessor workers."""

    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, state_kwargs=None, processor_class=TextProcessor):
        self._pool = _PatchedPool(
            processes, initializer, initargs, maxtasksperchild, state_kwargs,
            processor_class
        )

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


class TextTokenizer(TextProcessor):

    def process(self, text):
        return text.split()


if __name__ == "__main__":
    pool = TextProcessingPool(4, processor_class=TextTokenizer)
    texts = ['this is some test text for multiprocessing'] * 10

    results = pool.imap(texts)
    print(list(results))
