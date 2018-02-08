from calendar import timegm
from datetime import datetime, timedelta, date
from testfixtures.compat import new_class


@classmethod
def add(cls, *args, **kw):
    if 'tzinfo' in kw or len(args) > 7:
        raise TypeError('Cannot add using tzinfo on %s' % cls.__name__)
    if args and isinstance(args[0], cls.__bases__[0]):
        inst = args[0]
        tzinfo = getattr(inst, 'tzinfo', None)
        if tzinfo:
            if tzinfo != cls._tzta:
                raise ValueError(
                    'Cannot add %s with tzinfo of %s as configured to use %s' % (
                        inst.__class__.__name__, tzinfo, cls._tzta
                    ))
            inst = inst.replace(tzinfo=None)
        if cls._ct:
            inst = cls._ct(inst)
        cls._q.append(inst)
    else:
        cls._q.append(cls(*args, **kw))


@classmethod
def set_(cls, *args, **kw):
    if cls._q:
        cls._q = []
    cls.add(*args, **kw)


@classmethod
def tick(cls, *args, **kw):
    if kw:
        delta = timedelta(**kw)
    else:
        delta, = args
    cls._q[-1] += delta


def __add__(self, other):
    r = super(self.__class__, self).__add__(other)
    if self._ct:
        r = self._ct(r)
    return r


def __new__(cls, *args, **kw):
    if cls is cls._cls:
        return super(cls, cls).__new__(cls, *args, **kw)
    else:
        return cls._cls(*args, **kw)


@classmethod
def instantiate(cls):
    r = cls._q.pop(0)
    if not cls._q:
        cls._gap += cls._gap_d
        n = r + timedelta(**{cls._gap_t: cls._gap})
        if cls._ct:
            n = cls._ct(n)
        cls._q.append(n)
    return r


@classmethod
def now(cls, tz=None):
    r = cls._instantiate()
    if tz is not None:
        if cls._tzta:
            r = r - cls._tzta.utcoffset(r)
        r = tz.fromutc(r.replace(tzinfo=tz))
    return cls._ct(r)


@classmethod
def utcnow(cls):
    r = cls._instantiate()
    if cls._tzta is not None:
        r = r - cls._tzta.utcoffset(r)
    return r


def test_factory(n, type, default, args, kw, tz=None, **to_patch):
    q = []
    to_patch['_q'] = q
    to_patch['_tzta'] = tz
    to_patch['add'] = add
    to_patch['set'] = set_
    to_patch['tick'] = tick
    to_patch['__add__'] = __add__
    if '__new__' not in to_patch:
        to_patch['__new__'] = __new__
    class_ = new_class(n, (type, ), to_patch)
    strict = kw.pop('strict', False)
    if strict:
        class_._cls = class_
    else:
        class_._cls = type
    if args != (None, ):
        if not (args or kw):
            args = default
        class_.add(*args, **kw)
    return class_


def correct_date_method(self):
    return self._date_type(
        self.year,
        self.month,
        self.day
        )


@classmethod
def correct_datetime(cls, dt):
    return cls._cls(
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        dt.second,
        dt.microsecond,
        dt.tzinfo,
        )


def test_datetime(*args, **kw):
    if len(args) > 7:
        tz = args[7]
        args = args[:7]
    else:
        tz = kw.pop('tzinfo', getattr(args[0], 'tzinfo', None) if args else None)
    if 'delta' in kw:
        gap = kw.pop('delta')
        gap_delta = 0
    else:
        gap = 0
        gap_delta = 10
    delta_type = kw.pop('delta_type', 'seconds')
    date_type = kw.pop('date_type', date)
    return test_factory(
        'tdatetime', datetime, (2001, 1, 1, 0, 0, 0), args, kw, tz,
        _ct=correct_datetime,
        _instantiate=instantiate,
        now=now,
        utcnow=utcnow,
        _gap=gap,
        _gap_d=gap_delta,
        _gap_t=delta_type,
        date=correct_date_method,
        _date_type=date_type,
        )

test_datetime.__test__ = False


@classmethod
def correct_date(cls, d):
    return cls._cls(
        d.year,
        d.month,
        d.day,
        )


def test_date(*args, **kw):
    if 'delta' in kw:
        gap = kw.pop('delta')
        gap_delta = 0
    else:
        gap = 0
        gap_delta = 1
    delta_type = kw.pop('delta_type', 'days')
    return test_factory(
        'tdate', date, (2001, 1, 1), args, kw,
        _ct=correct_date,
        today=instantiate,
        _gap=gap,
        _gap_d=gap_delta,
        _gap_t=delta_type,
        )

ms = 10**6


def __time_new__(cls, *args, **kw):
    if args or kw:
        return super(cls, cls).__new__(cls, *args, **kw)
    else:
        val = cls.instantiate()
        t = timegm(val.utctimetuple())
        t += (float(val.microsecond)/ms)
        return t

test_date.__test__ = False


def test_time(*args, **kw):
    if 'tzinfo' in kw or len(args) > 7 or (args and getattr(args[0], 'tzinfo', None)):
        raise TypeError("You don't want to use tzinfo with test_time")
    if 'delta' in kw:
        gap = kw.pop('delta')
        gap_delta = 0
    else:
        gap = 0
        gap_delta = 1
    delta_type = kw.pop('delta_type', 'seconds')
    return test_factory(
        'ttime', datetime, (2001, 1, 1, 0, 0, 0), args, kw,
        _ct=None,
        instantiate=instantiate,
        _gap=gap,
        _gap_d=gap_delta,
        _gap_t=delta_type,
        __new__=__time_new__,
        )

test_time.__test__ = False
