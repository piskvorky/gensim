from __future__ import print_function
from logging import getLogger
from unittest import TestCase
from warnings import catch_warnings

from .mock import Mock

from testfixtures import Replacer, LogCapture, compare

root = getLogger()
one = getLogger('one')
two = getLogger('two')
child = getLogger('one.child')


class TestLogCapture(TestCase):

    def test_simple(self):
        root.info('before')
        l = LogCapture()
        root.info('during')
        l.uninstall()
        root.info('after')
        assert str(l) == "root INFO\n  during"

    def test_specific_logger(self):
        l = LogCapture('one')
        root.info('1')
        one.info('2')
        two.info('3')
        child.info('4')
        l.uninstall()
        assert str(l) == (
            "one INFO\n  2\n"
            "one.child INFO\n  4"
        )

    def test_multiple_loggers(self):
        l = LogCapture(('one.child','two'))
        root.info('1')
        one.info('2')
        two.info('3')
        child.info('4')
        l.uninstall()
        assert str(l) == (
            "two INFO\n  3\n"
            "one.child INFO\n  4"
        )

    def test_simple_manual_install(self):
        l = LogCapture(install=False)
        root.info('before')
        l.install()
        root.info('during')
        l.uninstall()
        root.info('after')
        assert str(l) == "root INFO\n  during"

    def test_uninstall(self):
        # Lets start off with a couple of loggers:

        root = getLogger()
        child = getLogger('child')

        # Lets also record the handlers for these loggers before
        # we start the test:

        before_root = root.handlers[:]
        before_child = child.handlers[:]

        # Lets also record the levels for the loggers:

        old_root_level=root.level
        old_child_level=child.level

        # Now the test:

        try:
            root.setLevel(49)
            child.setLevel(69)
            l1 = LogCapture()
            l2 = LogCapture('child')
            root = getLogger()
            root.info('1')
            child = getLogger('child')
            assert root.level == 1
            assert child.level == 1

            child.info('2')
            assert str(l1) == (
                "root INFO\n  1\n"
                "child INFO\n  2"
            )
            assert str(l2) == (
                "child INFO\n  2"
            )
            l2.uninstall()
            l1.uninstall()
            assert root.level == 49
            assert child.level == 69
        finally:
           root.setLevel(old_root_level)
           child.setLevel(old_child_level)

        # Now we check the handlers are as they were before
        # the test:
        assert root.handlers == before_root
        assert child.handlers == before_child

    def test_uninstall_all(self):
        before_handlers_root = root.handlers[:]
        before_handlers_child = child.handlers[:]

        l1 = LogCapture()
        l2 = LogCapture('one.child')

        # We can see that the LogCaptures have changed the
        # handlers, removing existing ones and installing
        # their own:

        assert len(root.handlers) == 1
        assert root.handlers != before_handlers_root
        assert len(child.handlers) == 1
        assert child.handlers != before_handlers_child

        # Now we show the function in action:

        LogCapture.uninstall_all()

        # ...and we can see the handlers are back as
        # they were beefore:

        assert before_handlers_root == root.handlers
        assert before_handlers_child == child.handlers

    def test_two_logcaptures_on_same_logger(self):
        # If you create more than one LogCapture on a single
        # logger, the 2nd one installed will stop the first
        # one working!

        l1 = LogCapture()
        root.info('1st message')
        assert str(l1) == "root INFO\n  1st message"
        l2 = LogCapture()
        root.info('2nd message')

        # So, l1 missed this message:
        assert str(l1) == "root INFO\n  1st message"

        # ...because l2 kicked it out and recorded the message:

        assert str(l2) == "root INFO\n  2nd message"

        LogCapture.uninstall_all()

    def test_uninstall_more_than_once(self):
        # There's no problem with uninstalling a LogCapture
        # more than once:

        old_level = root.level
        try:
           root.setLevel(49)
           l = LogCapture()
           assert root.level == 1
           l.uninstall()
           assert root.level == 49
           root.setLevel(69)
           l.uninstall()
           assert root.level == 69
        finally:
           root.setLevel(old_level)

        # And even when loggers have been uninstalled, there's
        # no problem having uninstall_all as a backstop:

        l.uninstall_all()

    def test_with_statement(self):
        root.info('before')
        with LogCapture() as l:
          root.info('during')
        root.info('after')
        assert str(l) == "root INFO\n  during"


class LogCaptureTests(TestCase):

    def test_remove_existing_handlers(self):
        logger = getLogger()
        # get original handlers
        original_handlers = logger.handlers
        # put in a stub which will blow up if used
        try:
            logger.handlers = start = [object()]

            with LogCapture() as l:
                logger.info('during')

            l.check(('root', 'INFO', 'during'))

            compare(logger.handlers, start)

        finally:
            # only executed if the test fails
            logger.handlers = original_handlers

    def test_atexit(self):
        # http://bugs.python.org/issue25532
        from .mock import call

        m = Mock()
        with Replacer() as r:
            # make sure the marker is false, other tests will
            # probably have set it
            r.replace('testfixtures.LogCapture.atexit_setup', False)
            r.replace('atexit.register', m.register)

            l = LogCapture()

            expected = [call.register(l.atexit)]

            compare(expected, m.mock_calls)

            with catch_warnings(record=True) as w:
                l.atexit()
                self.assertTrue(len(w), 1)
                compare(str(w[0].message), (
                    "LogCapture instances not uninstalled by shutdown, "
                    "loggers captured:\n"
                    "(None,)"
                    ))

            l.uninstall()

            compare(set(), LogCapture.instances)

            # check re-running has no ill effects
            l.atexit()

    def test_numeric_log_level(self):
        with LogCapture() as log:
            getLogger().log(42, 'running in the family')

        log.check(('root', 'Level 42', 'running in the family'))

    def test_enable_disabled_logger(self):
        logger = getLogger('disabled')
        logger.disabled = True
        with LogCapture('disabled') as log:
            logger.info('a log message')
        log.check(('disabled', 'INFO', 'a log message'))
        compare(logger.disabled, True)

    def test_no_propogate(self):
        logger = getLogger('child')
        # paranoid check
        compare(logger.propagate, True)
        with LogCapture() as global_log:
            with LogCapture('child', propagate=False) as child_log:
                logger.info('a log message')
                child_log.check(('child', 'INFO', 'a log message'))
        global_log.check()
        compare(logger.propagate, True)
