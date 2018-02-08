# -*- coding: utf-8 -*-
"""
This module offers a generic date/time string parser which is able to parse
most known formats to represent a date and/or time.

This module attempts to be forgiving with regards to unlikely input formats,
returning a datetime object even for dates which are ambiguous. If an element
of a date/time stamp is omitted, the following rules are applied:
- If AM or PM is left unspecified, a 24-hour clock is assumed, however, an hour
  on a 12-hour clock (``0 <= hour <= 12``) *must* be specified if AM or PM is
  specified.
- If a time zone is omitted, a timezone-naive datetime is returned.

If any other elements are missing, they are taken from the
:class:`datetime.datetime` object passed to the parameter ``default``. If this
results in a day number exceeding the valid number of days per month, the
value falls back to the end of the month.

Additional resources about date/time string formats can be found below:

- `A summary of the international standard date and time notation
  <http://www.cl.cam.ac.uk/~mgk25/iso-time.html>`_
- `W3C Date and Time Formats <http://www.w3.org/TR/NOTE-datetime>`_
- `Time Formats (Planetary Rings Node) <http://pds-rings.seti.org/tools/time_formats.html>`_
- `CPAN ParseDate module
  <http://search.cpan.org/~muir/Time-modules-2013.0912/lib/Time/ParseDate.pm>`_
- `Java SimpleDateFormat Class
  <https://docs.oracle.com/javase/6/docs/api/java/text/SimpleDateFormat.html>`_
"""
from __future__ import unicode_literals

import datetime
import string
import time
import collections
import re
from io import StringIO
from calendar import monthrange

from six import text_type, binary_type, integer_types

from . import relativedelta
from . import tz

__all__ = ["parse", "parserinfo"]


class _timelex(object):
    # Fractional seconds are sometimes split by a comma
    _split_decimal = re.compile("([.,])")

    def __init__(self, instream):
        if isinstance(instream, binary_type):
            instream = instream.decode()

        if isinstance(instream, text_type):
            instream = StringIO(instream)

        if getattr(instream, 'read', None) is None:
            raise TypeError('Parser must be a string or character stream, not '
                            '{itype}'.format(itype=instream.__class__.__name__))

        self.instream = instream
        self.charstack = []
        self.tokenstack = []
        self.eof = False

    def get_token(self):
        """
        This function breaks the time string into lexical units (tokens), which
        can be parsed by the parser. Lexical units are demarcated by changes in
        the character set, so any continuous string of letters is considered
        one unit, any continuous string of numbers is considered one unit.

        The main complication arises from the fact that dots ('.') can be used
        both as separators (e.g. "Sep.20.2009") or decimal points (e.g.
        "4:30:21.447"). As such, it is necessary to read the full context of
        any dot-separated strings before breaking it into tokens; as such, this
        function maintains a "token stack", for when the ambiguous context
        demands that multiple tokens be parsed at once.
        """
        if self.tokenstack:
            return self.tokenstack.pop(0)

        seenletters = False
        token = None
        state = None

        while not self.eof:
            # We only realize that we've reached the end of a token when we
            # find a character that's not part of the current token - since
            # that character may be part of the next token, it's stored in the
            # charstack.
            if self.charstack:
                nextchar = self.charstack.pop(0)
            else:
                nextchar = self.instream.read(1)
                while nextchar == '\x00':
                    nextchar = self.instream.read(1)

            if not nextchar:
                self.eof = True
                break
            elif not state:
                # First character of the token - determines if we're starting
                # to parse a word, a number or something else.
                token = nextchar
                if self.isword(nextchar):
                    state = 'a'
                elif self.isnum(nextchar):
                    state = '0'
                elif self.isspace(nextchar):
                    token = ' '
                    break  # emit token
                else:
                    break  # emit token
            elif state == 'a':
                # If we've already started reading a word, we keep reading
                # letters until we find something that's not part of a word.
                seenletters = True
                if self.isword(nextchar):
                    token += nextchar
                elif nextchar == '.':
                    token += nextchar
                    state = 'a.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token
            elif state == '0':
                # If we've already started reading a number, we keep reading
                # numbers until we find something that doesn't fit.
                if self.isnum(nextchar):
                    token += nextchar
                elif nextchar == '.' or (nextchar == ',' and len(token) >= 2):
                    token += nextchar
                    state = '0.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token
            elif state == 'a.':
                # If we've seen some letters and a dot separator, continue
                # parsing, and the tokens will be broken up later.
                seenletters = True
                if nextchar == '.' or self.isword(nextchar):
                    token += nextchar
                elif self.isnum(nextchar) and token[-1] == '.':
                    token += nextchar
                    state = '0.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token
            elif state == '0.':
                # If we've seen at least one dot separator, keep going, we'll
                # break up the tokens later.
                if nextchar == '.' or self.isnum(nextchar):
                    token += nextchar
                elif self.isword(nextchar) and token[-1] == '.':
                    token += nextchar
                    state = 'a.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token

        if (state in ('a.', '0.') and (seenletters or token.count('.') > 1 or
                                       token[-1] in '.,')):
            l = self._split_decimal.split(token)
            token = l[0]
            for tok in l[1:]:
                if tok:
                    self.tokenstack.append(tok)

        if state == '0.' and token.count('.') == 0:
            token = token.replace(',', '.')

        return token

    def __iter__(self):
        return self

    def __next__(self):
        token = self.get_token()
        if token is None:
            raise StopIteration

        return token

    def next(self):
        return self.__next__()  # Python 2.x support

    @classmethod
    def split(cls, s):
        return list(cls(s))

    @classmethod
    def isword(cls, nextchar):
        """ Whether or not the next character is part of a word """
        return nextchar.isalpha()

    @classmethod
    def isnum(cls, nextchar):
        """ Whether the next character is part of a number """
        return nextchar.isdigit()

    @classmethod
    def isspace(cls, nextchar):
        """ Whether the next character is whitespace """
        return nextchar.isspace()


class _resultbase(object):

    def __init__(self):
        for attr in self.__slots__:
            setattr(self, attr, None)

    def _repr(self, classname):
        l = []
        for attr in self.__slots__:
            value = getattr(self, attr)
            if value is not None:
                l.append("%s=%s" % (attr, repr(value)))
        return "%s(%s)" % (classname, ", ".join(l))

    def __len__(self):
        return (sum(getattr(self, attr) is not None
                    for attr in self.__slots__))

    def __repr__(self):
        return self._repr(self.__class__.__name__)


class parserinfo(object):
    """
    Class which handles what inputs are accepted. Subclass this to customize
    the language and acceptable values for each parameter.

    :param dayfirst:
            Whether to interpret the first value in an ambiguous 3-integer date
            (e.g. 01/05/09) as the day (``True``) or month (``False``). If
            ``yearfirst`` is set to ``True``, this distinguishes between YDM
            and YMD. Default is ``False``.

    :param yearfirst:
            Whether to interpret the first value in an ambiguous 3-integer date
            (e.g. 01/05/09) as the year. If ``True``, the first number is taken
            to be the year, otherwise the last number is taken to be the year.
            Default is ``False``.
    """

    # m from a.m/p.m, t from ISO T separator
    JUMP = [" ", ".", ",", ";", "-", "/", "'",
            "at", "on", "and", "ad", "m", "t", "of",
            "st", "nd", "rd", "th"]

    WEEKDAYS = [("Mon", "Monday"),
                ("Tue", "Tuesday"),
                ("Wed", "Wednesday"),
                ("Thu", "Thursday"),
                ("Fri", "Friday"),
                ("Sat", "Saturday"),
                ("Sun", "Sunday")]
    MONTHS = [("Jan", "January"),
              ("Feb", "February"),
              ("Mar", "March"),
              ("Apr", "April"),
              ("May", "May"),
              ("Jun", "June"),
              ("Jul", "July"),
              ("Aug", "August"),
              ("Sep", "Sept", "September"),
              ("Oct", "October"),
              ("Nov", "November"),
              ("Dec", "December")]
    HMS = [("h", "hour", "hours"),
           ("m", "minute", "minutes"),
           ("s", "second", "seconds")]
    AMPM = [("am", "a"),
            ("pm", "p")]
    UTCZONE = ["UTC", "GMT", "Z"]
    PERTAIN = ["of"]
    TZOFFSET = {}

    def __init__(self, dayfirst=False, yearfirst=False):
        self._jump = self._convert(self.JUMP)
        self._weekdays = self._convert(self.WEEKDAYS)
        self._months = self._convert(self.MONTHS)
        self._hms = self._convert(self.HMS)
        self._ampm = self._convert(self.AMPM)
        self._utczone = self._convert(self.UTCZONE)
        self._pertain = self._convert(self.PERTAIN)

        self.dayfirst = dayfirst
        self.yearfirst = yearfirst

        self._year = time.localtime().tm_year
        self._century = self._year // 100 * 100

    def _convert(self, lst):
        dct = {}
        for i, v in enumerate(lst):
            if isinstance(v, tuple):
                for v in v:
                    dct[v.lower()] = i
            else:
                dct[v.lower()] = i
        return dct

    def jump(self, name):
        return name.lower() in self._jump

    def weekday(self, name):
        if len(name) >= min(len(n) for n in self._weekdays.keys()):
            try:
                return self._weekdays[name.lower()]
            except KeyError:
                pass
        return None

    def month(self, name):
        if len(name) >= min(len(n) for n in self._months.keys()):
            try:
                return self._months[name.lower()] + 1
            except KeyError:
                pass
        return None

    def hms(self, name):
        try:
            return self._hms[name.lower()]
        except KeyError:
            return None

    def ampm(self, name):
        try:
            return self._ampm[name.lower()]
        except KeyError:
            return None

    def pertain(self, name):
        return name.lower() in self._pertain

    def utczone(self, name):
        return name.lower() in self._utczone

    def tzoffset(self, name):
        if name in self._utczone:
            return 0

        return self.TZOFFSET.get(name)

    def convertyear(self, year, century_specified=False):
        if year < 100 and not century_specified:
            year += self._century
            if abs(year - self._year) >= 50:
                if year < self._year:
                    year += 100
                else:
                    year -= 100
        return year

    def validate(self, res):
        # move to info
        if res.year is not None:
            res.year = self.convertyear(res.year, res.century_specified)

        if res.tzoffset == 0 and not res.tzname or res.tzname == 'Z':
            res.tzname = "UTC"
            res.tzoffset = 0
        elif res.tzoffset != 0 and res.tzname and self.utczone(res.tzname):
            res.tzoffset = 0
        return True


class _ymd(list):
    def __init__(self, tzstr, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.century_specified = False
        self.tzstr = tzstr

    @staticmethod
    def token_could_be_year(token, year):
        try:
            return int(token) == year
        except ValueError:
            return False

    @staticmethod
    def find_potential_year_tokens(year, tokens):
        return [token for token in tokens if _ymd.token_could_be_year(token, year)]

    def find_probable_year_index(self, tokens):
        """
        attempt to deduce if a pre 100 year was lost
         due to padded zeros being taken off
        """
        for index, token in enumerate(self):
            potential_year_tokens = _ymd.find_potential_year_tokens(token, tokens)
            if len(potential_year_tokens) == 1 and len(potential_year_tokens[0]) > 2:
                return index

    def append(self, val):
        if hasattr(val, '__len__'):
            if val.isdigit() and len(val) > 2:
                self.century_specified = True
        elif val > 100:
            self.century_specified = True

        super(self.__class__, self).append(int(val))

    def resolve_ymd(self, mstridx, yearfirst, dayfirst):
        len_ymd = len(self)
        year, month, day = (None, None, None)

        if len_ymd > 3:
            raise ValueError("More than three YMD values")
        elif len_ymd == 1 or (mstridx != -1 and len_ymd == 2):
            # One member, or two members with a month string
            if mstridx != -1:
                month = self[mstridx]
                del self[mstridx]

            if len_ymd > 1 or mstridx == -1:
                if self[0] > 31:
                    year = self[0]
                else:
                    day = self[0]

        elif len_ymd == 2:
            # Two members with numbers
            if self[0] > 31:
                # 99-01
                year, month = self
            elif self[1] > 31:
                # 01-99
                month, year = self
            elif dayfirst and self[1] <= 12:
                # 13-01
                day, month = self
            else:
                # 01-13
                month, day = self

        elif len_ymd == 3:
            # Three members
            if mstridx == 0:
                month, day, year = self
            elif mstridx == 1:
                if self[0] > 31 or (yearfirst and self[2] <= 31):
                    # 99-Jan-01
                    year, month, day = self
                else:
                    # 01-Jan-01
                    # Give precendence to day-first, since
                    # two-digit years is usually hand-written.
                    day, month, year = self

            elif mstridx == 2:
                # WTF!?
                if self[1] > 31:
                    # 01-99-Jan
                    day, year, month = self
                else:
                    # 99-01-Jan
                    year, day, month = self

            else:
                if self[0] > 31 or \
                    self.find_probable_year_index(_timelex.split(self.tzstr)) == 0 or \
                   (yearfirst and self[1] <= 12 and self[2] <= 31):
                    # 99-01-01
                    if dayfirst and self[2] <= 12:
                        year, day, month = self
                    else:
                        year, month, day = self
                elif self[0] > 12 or (dayfirst and self[1] <= 12):
                    # 13-01-01
                    day, month, year = self
                else:
                    # 01-13-01
                    month, day, year = self

        return year, month, day


class parser(object):
    def __init__(self, info=None):
        self.info = info or parserinfo()

    def parse(self, timestr, default=None, ignoretz=False, tzinfos=None, **kwargs):
        """
        Parse the date/time string into a :class:`datetime.datetime` object.

        :param timestr:
            Any date/time string using the supported formats.

        :param default:
            The default datetime object, if this is a datetime object and not
            ``None``, elements specified in ``timestr`` replace elements in the
            default object.

        :param ignoretz:
            If set ``True``, time zones in parsed strings are ignored and a
            naive :class:`datetime.datetime` object is returned.

        :param tzinfos:
            Additional time zone names / aliases which may be present in the
            string. This argument maps time zone names (and optionally offsets
            from those time zones) to time zones. This parameter can be a
            dictionary with timezone aliases mapping time zone names to time
            zones or a function taking two parameters (``tzname`` and
            ``tzoffset``) and returning a time zone.

            The timezones to which the names are mapped can be an integer
            offset from UTC in minutes or a :class:`tzinfo` object.

            .. doctest::
               :options: +NORMALIZE_WHITESPACE

                >>> from dateutil.parser import parse
                >>> from dateutil.tz import gettz
                >>> tzinfos = {"BRST": -10800, "CST": gettz("America/Chicago")}
                >>> parse("2012-01-19 17:21:00 BRST", tzinfos=tzinfos)
                datetime.datetime(2012, 1, 19, 17, 21, tzinfo=tzoffset(u'BRST', -10800))
                >>> parse("2012-01-19 17:21:00 CST", tzinfos=tzinfos)
                datetime.datetime(2012, 1, 19, 17, 21,
                                  tzinfo=tzfile('/usr/share/zoneinfo/America/Chicago'))

            This parameter is ignored if ``ignoretz`` is set.

        :param **kwargs:
            Keyword arguments as passed to ``_parse()``.

        :return:
            Returns a :class:`datetime.datetime` object or, if the
            ``fuzzy_with_tokens`` option is ``True``, returns a tuple, the
            first element being a :class:`datetime.datetime` object, the second
            a tuple containing the fuzzy tokens.

        :raises ValueError:
            Raised for invalid or unknown string format, if the provided
            :class:`tzinfo` is not in a valid format, or if an invalid date
            would be created.

        :raises TypeError:
            Raised for non-string or character stream input.

        :raises OverflowError:
            Raised if the parsed date exceeds the largest valid C integer on
            your system.
        """

        if default is None:
            default = datetime.datetime.now().replace(hour=0, minute=0,
                                                      second=0, microsecond=0)

        res, skipped_tokens = self._parse(timestr, **kwargs)

        if res is None:
            raise ValueError("Unknown string format")

        if len(res) == 0:
            raise ValueError("String does not contain a date.")

        repl = {}
        for attr in ("year", "month", "day", "hour",
                     "minute", "second", "microsecond"):
            value = getattr(res, attr)
            if value is not None:
                repl[attr] = value

        if 'day' not in repl:
            # If the default day exceeds the last day of the month, fall back to
            # the end of the month.
            cyear = default.year if res.year is None else res.year
            cmonth = default.month if res.month is None else res.month
            cday = default.day if res.day is None else res.day

            if cday > monthrange(cyear, cmonth)[1]:
                repl['day'] = monthrange(cyear, cmonth)[1]

        ret = default.replace(**repl)

        if res.weekday is not None and not res.day:
            ret = ret+relativedelta.relativedelta(weekday=res.weekday)

        if not ignoretz:
            if (isinstance(tzinfos, collections.Callable) or
                    tzinfos and res.tzname in tzinfos):

                if isinstance(tzinfos, collections.Callable):
                    tzdata = tzinfos(res.tzname, res.tzoffset)
                else:
                    tzdata = tzinfos.get(res.tzname)

                if isinstance(tzdata, datetime.tzinfo):
                    tzinfo = tzdata
                elif isinstance(tzdata, text_type):
                    tzinfo = tz.tzstr(tzdata)
                elif isinstance(tzdata, integer_types):
                    tzinfo = tz.tzoffset(res.tzname, tzdata)
                else:
                    raise ValueError("Offset must be tzinfo subclass, "
                                     "tz string, or int offset.")
                ret = ret.replace(tzinfo=tzinfo)
            elif res.tzname and res.tzname in time.tzname:
                ret = ret.replace(tzinfo=tz.tzlocal())
            elif res.tzoffset == 0:
                ret = ret.replace(tzinfo=tz.tzutc())
            elif res.tzoffset:
                ret = ret.replace(tzinfo=tz.tzoffset(res.tzname, res.tzoffset))

        if kwargs.get('fuzzy_with_tokens', False):
            return ret, skipped_tokens
        else:
            return ret

    class _result(_resultbase):
        __slots__ = ["year", "month", "day", "weekday",
                     "hour", "minute", "second", "microsecond",
                     "tzname", "tzoffset", "ampm"]

    def _parse(self, timestr, dayfirst=None, yearfirst=None, fuzzy=False,
               fuzzy_with_tokens=False):
        """
        Private method which performs the heavy lifting of parsing, called from
        ``parse()``, which passes on its ``kwargs`` to this function.

        :param timestr:
            The string to parse.

        :param dayfirst:
            Whether to interpret the first value in an ambiguous 3-integer date
            (e.g. 01/05/09) as the day (``True``) or month (``False``). If
            ``yearfirst`` is set to ``True``, this distinguishes between YDM
            and YMD. If set to ``None``, this value is retrieved from the
            current :class:`parserinfo` object (which itself defaults to
            ``False``).

        :param yearfirst:
            Whether to interpret the first value in an ambiguous 3-integer date
            (e.g. 01/05/09) as the year. If ``True``, the first number is taken
            to be the year, otherwise the last number is taken to be the year.
            If this is set to ``None``, the value is retrieved from the current
            :class:`parserinfo` object (which itself defaults to ``False``).

        :param fuzzy:
            Whether to allow fuzzy parsing, allowing for string like "Today is
            January 1, 2047 at 8:21:00AM".

        :param fuzzy_with_tokens:
            If ``True``, ``fuzzy`` is automatically set to True, and the parser
            will return a tuple where the first element is the parsed
            :class:`datetime.datetime` datetimestamp and the second element is
            a tuple containing the portions of the string which were ignored:

            .. doctest::

                >>> from dateutil.parser import parse
                >>> parse("Today is January 1, 2047 at 8:21:00AM", fuzzy_with_tokens=True)
                (datetime.datetime(2047, 1, 1, 8, 21), (u'Today is ', u' ', u'at '))

        """
        if fuzzy_with_tokens:
            fuzzy = True

        info = self.info

        if dayfirst is None:
            dayfirst = info.dayfirst

        if yearfirst is None:
            yearfirst = info.yearfirst

        res = self._result()
        l = _timelex.split(timestr)         # Splits the timestr into tokens

        # keep up with the last token skipped so we can recombine
        # consecutively skipped tokens (-2 for when i begins at 0).
        last_skipped_token_i = -2
        skipped_tokens = list()

        try:
            # year/month/day list
            ymd = _ymd(timestr)

            # Index of the month string in ymd
            mstridx = -1

            len_l = len(l)
            i = 0
            while i < len_l:

                # Check if it's a number
                try:
                    value_repr = l[i]
                    value = float(value_repr)
                except ValueError:
                    value = None

                if value is not None:
                    # Token is a number
                    len_li = len(l[i])
                    i += 1

                    if (len(ymd) == 3 and len_li in (2, 4)
                        and res.hour is None and (i >= len_l or (l[i] != ':' and
                                                  info.hms(l[i]) is None))):
                        # 19990101T23[59]
                        s = l[i-1]
                        res.hour = int(s[:2])

                        if len_li == 4:
                            res.minute = int(s[2:])

                    elif len_li == 6 or (len_li > 6 and l[i-1].find('.') == 6):
                        # YYMMDD or HHMMSS[.ss]
                        s = l[i-1]

                        if not ymd and l[i-1].find('.') == -1:
                            #ymd.append(info.convertyear(int(s[:2])))

                            ymd.append(s[:2])
                            ymd.append(s[2:4])
                            ymd.append(s[4:])
                        else:
                            # 19990101T235959[.59]
                            res.hour = int(s[:2])
                            res.minute = int(s[2:4])
                            res.second, res.microsecond = _parsems(s[4:])

                    elif len_li in (8, 12, 14):
                        # YYYYMMDD
                        s = l[i-1]
                        ymd.append(s[:4])
                        ymd.append(s[4:6])
                        ymd.append(s[6:8])

                        if len_li > 8:
                            res.hour = int(s[8:10])
                            res.minute = int(s[10:12])

                            if len_li > 12:
                                res.second = int(s[12:])

                    elif ((i < len_l and info.hms(l[i]) is not None) or
                          (i+1 < len_l and l[i] == ' ' and
                           info.hms(l[i+1]) is not None)):

                        # HH[ ]h or MM[ ]m or SS[.ss][ ]s
                        if l[i] == ' ':
                            i += 1

                        idx = info.hms(l[i])

                        while True:
                            if idx == 0:
                                res.hour = int(value)

                                if value % 1:
                                    res.minute = int(60*(value % 1))

                            elif idx == 1:
                                res.minute = int(value)

                                if value % 1:
                                    res.second = int(60*(value % 1))

                            elif idx == 2:
                                res.second, res.microsecond = \
                                    _parsems(value_repr)

                            i += 1

                            if i >= len_l or idx == 2:
                                break

                            # 12h00
                            try:
                                value_repr = l[i]
                                value = float(value_repr)
                            except ValueError:
                                break
                            else:
                                i += 1
                                idx += 1

                                if i < len_l:
                                    newidx = info.hms(l[i])

                                    if newidx is not None:
                                        idx = newidx

                    elif (i == len_l and l[i-2] == ' ' and
                          info.hms(l[i-3]) is not None):
                        # X h MM or X m SS
                        idx = info.hms(l[i-3])

                        if idx == 0:               # h
                            res.minute = int(value)

                            sec_remainder = value % 1
                            if sec_remainder:
                                res.second = int(60 * sec_remainder)
                        elif idx == 1:             # m
                            res.second, res.microsecond = \
                                _parsems(value_repr)

                        # We don't need to advance the tokens here because the
                        # i == len_l call indicates that we're looking at all
                        # the tokens already.

                    elif i+1 < len_l and l[i] == ':':
                        # HH:MM[:SS[.ss]]
                        res.hour = int(value)
                        i += 1
                        value = float(l[i])
                        res.minute = int(value)

                        if value % 1:
                            res.second = int(60*(value % 1))

                        i += 1

                        if i < len_l and l[i] == ':':
                            res.second, res.microsecond = _parsems(l[i+1])
                            i += 2

                    elif i < len_l and l[i] in ('-', '/', '.'):
                        sep = l[i]
                        ymd.append(value_repr)
                        i += 1

                        if i < len_l and not info.jump(l[i]):
                            try:
                                # 01-01[-01]
                                ymd.append(l[i])
                            except ValueError:
                                # 01-Jan[-01]
                                value = info.month(l[i])

                                if value is not None:
                                    ymd.append(value)
                                    assert mstridx == -1
                                    mstridx = len(ymd)-1
                                else:
                                    return None, None

                            i += 1

                            if i < len_l and l[i] == sep:
                                # We have three members
                                i += 1
                                value = info.month(l[i])

                                if value is not None:
                                    ymd.append(value)
                                    mstridx = len(ymd)-1
                                    assert mstridx == -1
                                else:
                                    ymd.append(l[i])

                                i += 1
                    elif i >= len_l or info.jump(l[i]):
                        if i+1 < len_l and info.ampm(l[i+1]) is not None:
                            # 12 am
                            res.hour = int(value)

                            if res.hour < 12 and info.ampm(l[i+1]) == 1:
                                res.hour += 12
                            elif res.hour == 12 and info.ampm(l[i+1]) == 0:
                                res.hour = 0

                            i += 1
                        else:
                            # Year, month or day
                            ymd.append(value)
                        i += 1
                    elif info.ampm(l[i]) is not None:

                        # 12am
                        res.hour = int(value)

                        if res.hour < 12 and info.ampm(l[i]) == 1:
                            res.hour += 12
                        elif res.hour == 12 and info.ampm(l[i]) == 0:
                            res.hour = 0
                        i += 1

                    elif not fuzzy:
                        return None, None
                    else:
                        i += 1
                    continue

                # Check weekday
                value = info.weekday(l[i])
                if value is not None:
                    res.weekday = value
                    i += 1
                    continue

                # Check month name
                value = info.month(l[i])
                if value is not None:
                    ymd.append(value)
                    assert mstridx == -1
                    mstridx = len(ymd)-1

                    i += 1
                    if i < len_l:
                        if l[i] in ('-', '/'):
                            # Jan-01[-99]
                            sep = l[i]
                            i += 1
                            ymd.append(l[i])
                            i += 1

                            if i < len_l and l[i] == sep:
                                # Jan-01-99
                                i += 1
                                ymd.append(l[i])
                                i += 1

                        elif (i+3 < len_l and l[i] == l[i+2] == ' '
                              and info.pertain(l[i+1])):
                            # Jan of 01
                            # In this case, 01 is clearly year
                            try:
                                value = int(l[i+3])
                            except ValueError:
                                # Wrong guess
                                pass
                            else:
                                # Convert it here to become unambiguous
                                ymd.append(str(info.convertyear(value)))
                            i += 4
                    continue

                # Check am/pm
                value = info.ampm(l[i])
                if value is not None:
                    # For fuzzy parsing, 'a' or 'am' (both valid English words)
                    # may erroneously trigger the AM/PM flag. Deal with that
                    # here.
                    val_is_ampm = True

                    # If there's already an AM/PM flag, this one isn't one.
                    if fuzzy and res.ampm is not None:
                        val_is_ampm = False

                    # If AM/PM is found and hour is not, raise a ValueError
                    if res.hour is None:
                        if fuzzy:
                            val_is_ampm = False
                        else:
                            raise ValueError('No hour specified with ' +
                                             'AM or PM flag.')
                    elif not 0 <= res.hour <= 12:
                        # If AM/PM is found, it's a 12 hour clock, so raise
                        # an error for invalid range
                        if fuzzy:
                            val_is_ampm = False
                        else:
                            raise ValueError('Invalid hour specified for ' +
                                             '12-hour clock.')

                    if val_is_ampm:
                        if value == 1 and res.hour < 12:
                            res.hour += 12
                        elif value == 0 and res.hour == 12:
                            res.hour = 0

                        res.ampm = value

                    elif fuzzy:
                        last_skipped_token_i = self._skip_token(skipped_tokens,
                                                    last_skipped_token_i, i, l)
                    i += 1
                    continue

                # Check for a timezone name
                if (res.hour is not None and len(l[i]) <= 5 and
                        res.tzname is None and res.tzoffset is None and
                        not [x for x in l[i] if x not in
                             string.ascii_uppercase]):
                    res.tzname = l[i]
                    res.tzoffset = info.tzoffset(res.tzname)
                    i += 1

                    # Check for something like GMT+3, or BRST+3. Notice
                    # that it doesn't mean "I am 3 hours after GMT", but
                    # "my time +3 is GMT". If found, we reverse the
                    # logic so that timezone parsing code will get it
                    # right.
                    if i < len_l and l[i] in ('+', '-'):
                        l[i] = ('+', '-')[l[i] == '+']
                        res.tzoffset = None
                        if info.utczone(res.tzname):
                            # With something like GMT+3, the timezone
                            # is *not* GMT.
                            res.tzname = None

                    continue

                # Check for a numbered timezone
                if res.hour is not None and l[i] in ('+', '-'):
                    signal = (-1, 1)[l[i] == '+']
                    i += 1
                    len_li = len(l[i])

                    if len_li == 4:
                        # -0300
                        res.tzoffset = int(l[i][:2])*3600+int(l[i][2:])*60
                    elif i+1 < len_l and l[i+1] == ':':
                        # -03:00
                        res.tzoffset = int(l[i])*3600+int(l[i+2])*60
                        i += 2
                    elif len_li <= 2:
                        # -[0]3
                        res.tzoffset = int(l[i][:2])*3600
                    else:
                        return None, None
                    i += 1

                    res.tzoffset *= signal

                    # Look for a timezone name between parenthesis
                    if (i+3 < len_l and
                        info.jump(l[i]) and l[i+1] == '(' and l[i+3] == ')' and
                        3 <= len(l[i+2]) <= 5 and
                        not [x for x in l[i+2]
                             if x not in string.ascii_uppercase]):
                        # -0300 (BRST)
                        res.tzname = l[i+2]
                        i += 4
                    continue

                # Check jumps
                if not (info.jump(l[i]) or fuzzy):
                    return None, None

                last_skipped_token_i = self._skip_token(skipped_tokens,
                                                last_skipped_token_i, i, l)
                i += 1

            # Process year/month/day
            year, month, day = ymd.resolve_ymd(mstridx, yearfirst, dayfirst)
            if year is not None:
                res.year = year
                res.century_specified = ymd.century_specified

            if month is not None:
                res.month = month

            if day is not None:
                res.day = day

        except (IndexError, ValueError, AssertionError):
            return None, None

        if not info.validate(res):
            return None, None

        if fuzzy_with_tokens:
            return res, tuple(skipped_tokens)
        else:
            return res, None

    @staticmethod
    def _skip_token(skipped_tokens, last_skipped_token_i, i, l):
        if last_skipped_token_i == i - 1:
            # recombine the tokens
            skipped_tokens[-1] += l[i]
        else:
            # just append
            skipped_tokens.append(l[i])
        last_skipped_token_i = i
        return last_skipped_token_i


DEFAULTPARSER = parser()


def parse(timestr, parserinfo=None, **kwargs):
    """

    Parse a string in one of the supported formats, using the
    ``parserinfo`` parameters.

    :param timestr:
        A string containing a date/time stamp.

    :param parserinfo:
        A :class:`parserinfo` object containing parameters for the parser.
        If ``None``, the default arguments to the :class:`parserinfo`
        constructor are used.

    The ``**kwargs`` parameter takes the following keyword arguments:

    :param default:
        The default datetime object, if this is a datetime object and not
        ``None``, elements specified in ``timestr`` replace elements in the
        default object.

    :param ignoretz:
        If set ``True``, time zones in parsed strings are ignored and a naive
        :class:`datetime` object is returned.

    :param tzinfos:
            Additional time zone names / aliases which may be present in the
            string. This argument maps time zone names (and optionally offsets
            from those time zones) to time zones. This parameter can be a
            dictionary with timezone aliases mapping time zone names to time
            zones or a function taking two parameters (``tzname`` and
            ``tzoffset``) and returning a time zone.

            The timezones to which the names are mapped can be an integer
            offset from UTC in minutes or a :class:`tzinfo` object.

            .. doctest::
               :options: +NORMALIZE_WHITESPACE

                >>> from dateutil.parser import parse
                >>> from dateutil.tz import gettz
                >>> tzinfos = {"BRST": -10800, "CST": gettz("America/Chicago")}
                >>> parse("2012-01-19 17:21:00 BRST", tzinfos=tzinfos)
                datetime.datetime(2012, 1, 19, 17, 21, tzinfo=tzoffset(u'BRST', -10800))
                >>> parse("2012-01-19 17:21:00 CST", tzinfos=tzinfos)
                datetime.datetime(2012, 1, 19, 17, 21,
                                  tzinfo=tzfile('/usr/share/zoneinfo/America/Chicago'))

            This parameter is ignored if ``ignoretz`` is set.

    :param dayfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the day (``True``) or month (``False``). If
        ``yearfirst`` is set to ``True``, this distinguishes between YDM and
        YMD. If set to ``None``, this value is retrieved from the current
        :class:`parserinfo` object (which itself defaults to ``False``).

    :param yearfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the year. If ``True``, the first number is taken to
        be the year, otherwise the last number is taken to be the year. If
        this is set to ``None``, the value is retrieved from the current
        :class:`parserinfo` object (which itself defaults to ``False``).

    :param fuzzy:
        Whether to allow fuzzy parsing, allowing for string like "Today is
        January 1, 2047 at 8:21:00AM".

    :param fuzzy_with_tokens:
        If ``True``, ``fuzzy`` is automatically set to True, and the parser
        will return a tuple where the first element is the parsed
        :class:`datetime.datetime` datetimestamp and the second element is
        a tuple containing the portions of the string which were ignored:

        .. doctest::

            >>> from dateutil.parser import parse
            >>> parse("Today is January 1, 2047 at 8:21:00AM", fuzzy_with_tokens=True)
            (datetime.datetime(2047, 1, 1, 8, 21), (u'Today is ', u' ', u'at '))

    :return:
        Returns a :class:`datetime.datetime` object or, if the
        ``fuzzy_with_tokens`` option is ``True``, returns a tuple, the
        first element being a :class:`datetime.datetime` object, the second
        a tuple containing the fuzzy tokens.

    :raises ValueError:
        Raised for invalid or unknown string format, if the provided
        :class:`tzinfo` is not in a valid format, or if an invalid date
        would be created.

    :raises OverflowError:
        Raised if the parsed date exceeds the largest valid C integer on
        your system.
    """
    if parserinfo:
        return parser(parserinfo).parse(timestr, **kwargs)
    else:
        return DEFAULTPARSER.parse(timestr, **kwargs)


class _tzparser(object):

    class _result(_resultbase):

        __slots__ = ["stdabbr", "stdoffset", "dstabbr", "dstoffset",
                     "start", "end"]

        class _attr(_resultbase):
            __slots__ = ["month", "week", "weekday",
                         "yday", "jyday", "day", "time"]

        def __repr__(self):
            return self._repr("")

        def __init__(self):
            _resultbase.__init__(self)
            self.start = self._attr()
            self.end = self._attr()

    def parse(self, tzstr):
        res = self._result()
        l = _timelex.split(tzstr)
        try:

            len_l = len(l)

            i = 0
            while i < len_l:
                # BRST+3[BRDT[+2]]
                j = i
                while j < len_l and not [x for x in l[j]
                                         if x in "0123456789:,-+"]:
                    j += 1
                if j != i:
                    if not res.stdabbr:
                        offattr = "stdoffset"
                        res.stdabbr = "".join(l[i:j])
                    else:
                        offattr = "dstoffset"
                        res.dstabbr = "".join(l[i:j])
                    i = j
                    if (i < len_l and (l[i] in ('+', '-') or l[i][0] in
                                       "0123456789")):
                        if l[i] in ('+', '-'):
                            # Yes, that's right.  See the TZ variable
                            # documentation.
                            signal = (1, -1)[l[i] == '+']
                            i += 1
                        else:
                            signal = -1
                        len_li = len(l[i])
                        if len_li == 4:
                            # -0300
                            setattr(res, offattr, (int(l[i][:2])*3600 +
                                                   int(l[i][2:])*60)*signal)
                        elif i+1 < len_l and l[i+1] == ':':
                            # -03:00
                            setattr(res, offattr,
                                    (int(l[i])*3600+int(l[i+2])*60)*signal)
                            i += 2
                        elif len_li <= 2:
                            # -[0]3
                            setattr(res, offattr,
                                    int(l[i][:2])*3600*signal)
                        else:
                            return None
                        i += 1
                    if res.dstabbr:
                        break
                else:
                    break

            if i < len_l:
                for j in range(i, len_l):
                    if l[j] == ';':
                        l[j] = ','

                assert l[i] == ','

                i += 1

            if i >= len_l:
                pass
            elif (8 <= l.count(',') <= 9 and
                  not [y for x in l[i:] if x != ','
                       for y in x if y not in "0123456789"]):
                # GMT0BST,3,0,30,3600,10,0,26,7200[,3600]
                for x in (res.start, res.end):
                    x.month = int(l[i])
                    i += 2
                    if l[i] == '-':
                        value = int(l[i+1])*-1
                        i += 1
                    else:
                        value = int(l[i])
                    i += 2
                    if value:
                        x.week = value
                        x.weekday = (int(l[i])-1) % 7
                    else:
                        x.day = int(l[i])
                    i += 2
                    x.time = int(l[i])
                    i += 2
                if i < len_l:
                    if l[i] in ('-', '+'):
                        signal = (-1, 1)[l[i] == "+"]
                        i += 1
                    else:
                        signal = 1
                    res.dstoffset = (res.stdoffset+int(l[i]))*signal
            elif (l.count(',') == 2 and l[i:].count('/') <= 2 and
                  not [y for x in l[i:] if x not in (',', '/', 'J', 'M',
                                                     '.', '-', ':')
                       for y in x if y not in "0123456789"]):
                for x in (res.start, res.end):
                    if l[i] == 'J':
                        # non-leap year day (1 based)
                        i += 1
                        x.jyday = int(l[i])
                    elif l[i] == 'M':
                        # month[-.]week[-.]weekday
                        i += 1
                        x.month = int(l[i])
                        i += 1
                        assert l[i] in ('-', '.')
                        i += 1
                        x.week = int(l[i])
                        if x.week == 5:
                            x.week = -1
                        i += 1
                        assert l[i] in ('-', '.')
                        i += 1
                        x.weekday = (int(l[i])-1) % 7
                    else:
                        # year day (zero based)
                        x.yday = int(l[i])+1

                    i += 1

                    if i < len_l and l[i] == '/':
                        i += 1
                        # start time
                        len_li = len(l[i])
                        if len_li == 4:
                            # -0300
                            x.time = (int(l[i][:2])*3600+int(l[i][2:])*60)
                        elif i+1 < len_l and l[i+1] == ':':
                            # -03:00
                            x.time = int(l[i])*3600+int(l[i+2])*60
                            i += 2
                            if i+1 < len_l and l[i+1] == ':':
                                i += 2
                                x.time += int(l[i])
                        elif len_li <= 2:
                            # -[0]3
                            x.time = (int(l[i][:2])*3600)
                        else:
                            return None
                        i += 1

                    assert i == len_l or l[i] == ','

                    i += 1

                assert i >= len_l

        except (IndexError, ValueError, AssertionError):
            return None

        return res


DEFAULTTZPARSER = _tzparser()


def _parsetz(tzstr):
    return DEFAULTTZPARSER.parse(tzstr)


def _parsems(value):
    """Parse a I[.F] seconds value into (seconds, microseconds)."""
    if "." not in value:
        return int(value), 0
    else:
        i, f = value.split(".")
        return int(i), int(f.ljust(6, "0")[:6])


# vim:ts=4:sw=4:et
