import pkg_resources
import time
import warnings

import pytest

from _pytest.runner import runtestprotocol
from _pytest.resultlog import ResultLog


def works_with_current_xdist():
    """Returns compatibility with installed pytest-xdist version.

    When running tests in parallel using pytest-xdist < 1.20.0, the first
    report that is logged will finish and terminate the current node rather
    rerunning the test. Thus we must skip logging of intermediate results under
    these circumstances, otherwise no test is rerun.

    """
    try:
        d = pkg_resources.get_distribution('pytest-xdist')
        return d.parsed_version >= pkg_resources.parse_version('1.20')
    except pkg_resources.DistributionNotFound:
        return None


# command line options
def pytest_addoption(parser):
    group = parser.getgroup(
        "rerunfailures",
        "re-run failing tests to eliminate flaky failures")
    group._addoption(
        '--reruns',
        action="store",
        dest="reruns",
        type=int,
        default=0,
        help="number of times to re-run failed tests. defaults to 0.")
    group._addoption(
        '--reruns-delay',
        action='store',
        dest='reruns_delay',
        type=float,
        default=0,
        help='add time (seconds) delay between reruns.'
    )


def pytest_configure(config):
    # add flaky marker
    config.addinivalue_line(
        "markers", "flaky(reruns=1, reruns_delay=0): mark test to re-run up "
                   "to 'reruns' times. Add a delay of 'reruns_delay' seconds "
                   "between re-runs.")


# making sure the options make sense
# should run before / at the begining of pytest_cmdline_main
def check_options(config):
    val = config.getvalue
    if not val("collectonly"):
        if config.option.reruns != 0:
            if config.option.usepdb:   # a core option
                raise pytest.UsageError("--reruns incompatible with --pdb")

    resultlog = getattr(config, '_resultlog', None)
    if resultlog:
        logfile = resultlog.logfile
        config.pluginmanager.unregister(resultlog)
        config._resultlog = RerunResultLog(config, logfile)
        config.pluginmanager.register(config._resultlog)


def get_reruns_count(item):
    rerun_marker = item.get_marker("flaky")
    reruns = None

    # use the marker as a priority over the global setting.
    if rerun_marker is not None:
        if "reruns" in rerun_marker.kwargs:
            # check for keyword arguments
            reruns = rerun_marker.kwargs["reruns"]
        elif len(rerun_marker.args) > 0:
            # check for arguments
            reruns = rerun_marker.args[0]
        else:
            reruns = 1
    elif item.session.config.option.reruns:
        # default to the global setting
        reruns = item.session.config.option.reruns

    return reruns


def get_reruns_delay(item):
    rerun_marker = item.get_marker("flaky")

    if rerun_marker is not None:
        if "reruns_delay" in rerun_marker.kwargs:
            delay = rerun_marker.kwargs["reruns_delay"]
        elif len(rerun_marker.args) > 1:
            # check for arguments
            delay = rerun_marker.args[1]
        else:
            delay = 0
    else:
        delay = item.session.config.option.reruns_delay

    if delay < 0:
        delay = 0
        warnings.warn('Delay time between re-runs cannot be < 0. '
                      'Using default value: 0')

    return delay


def pytest_runtest_protocol(item, nextitem):
    """
    Note: when teardown fails, two reports are generated for the case, one for
    the test case and the other for the teardown error.
    """

    reruns = get_reruns_count(item)
    if reruns is None:
        # global setting is not specified, and this test is not marked with
        # flaky
        return

    # while this doesn't need to be run with every item, it will fail on the
    # first item if necessary
    check_options(item.session.config)
    delay = get_reruns_delay(item)
    parallel = hasattr(item.config, 'slaveinput')

    for i in range(reruns + 1):  # ensure at least one run of each item
        item.ihook.pytest_runtest_logstart(nodeid=item.nodeid,
                                           location=item.location)
        reports = runtestprotocol(item, nextitem=nextitem, log=False)

        for report in reports:  # 3 reports: setup, test, teardown
            report.rerun = i
            xfail = hasattr(report, 'wasxfail')
            if i == reruns or not report.failed or xfail:
                # last run or no failure detected, log normally
                item.ihook.pytest_runtest_logreport(report=report)
            else:
                # failure detected and reruns not exhausted, since i < reruns
                report.outcome = 'rerun'
                time.sleep(delay)

                if not parallel or works_with_current_xdist():
                    # will rerun test, log intermediate result
                    item.ihook.pytest_runtest_logreport(report=report)

                break  # trigger rerun
        else:
            return True  # no need to rerun

    return True


def pytest_report_teststatus(report):
    """Adapted from https://pytest.org/latest/_modules/_pytest/skipping.html
    """
    if report.outcome == 'rerun':
        return 'rerun', 'R', ('RERUN', {'yellow': True})


def pytest_terminal_summary(terminalreporter):
    """Adapted from https://pytest.org/latest/_modules/_pytest/skipping.html
    """
    tr = terminalreporter
    if not tr.reportchars:
        return

    lines = []
    for char in tr.reportchars:
        if char in 'rR':
            show_rerun(terminalreporter, lines)

    if lines:
        tr._tw.sep("=", "rerun test summary info")
        for line in lines:
            tr._tw.line(line)


def show_rerun(terminalreporter, lines):
    rerun = terminalreporter.stats.get("rerun")
    if rerun:
        for rep in rerun:
            pos = rep.nodeid
            lines.append("RERUN %s" % (pos,))


class RerunResultLog(ResultLog):
    def __init__(self, config, logfile):
        ResultLog.__init__(self, config, logfile)

    def pytest_runtest_logreport(self, report):
        """
        Adds support for rerun report fix for issue:
        https://github.com/pytest-dev/pytest-rerunfailures/issues/28
        """
        if report.when != "call" and report.passed:
            return
        res = self.config.hook.pytest_report_teststatus(report=report)
        code = res[1]
        if code == 'x':
            longrepr = str(report.longrepr)
        elif code == 'X':
            longrepr = ''
        elif report.passed:
            longrepr = ""
        elif report.failed:
            longrepr = str(report.longrepr)
        elif report.skipped:
            longrepr = str(report.longrepr[2])
        elif report.outcome == 'rerun':
            longrepr = str(report.longrepr)

        self.log_outcome(report, code, longrepr)
