from unittest.mock import patch
from types import SimpleNamespace
import pytest
from eesd.recursive_execution import run_test,feedback_for_tests,Outcome


def test_fail_closed_without_bwrap():
    with patch('eesd.recursive_execution.shutil.which',return_value=None):
        with pytest.raises(RuntimeError,match='bubblewrap required'):
            run_test('print(1)',{'input':'','expected':'1'})


def test_valid_stdout_comparison_and_no_network_mount():
    with patch('eesd.recursive_execution.shutil.which',return_value='/usr/bin/bwrap'),\
         patch('eesd.recursive_execution.subprocess.run',return_value=SimpleNamespace(stdout='3\n',stderr='',returncode=0)) as mocked:
        result=run_test('print(3)',{'input':'','expected':'3\n'})
        assert result.status=='PASS'
        cmd=mocked.call_args.args[0]
        assert '--unshare-all' in cmd and '--bind' not in cmd
        assert 'python3' in cmd[-5]


def test_bwrap_startup_errors_are_fatal():
    with patch('eesd.recursive_execution.shutil.which',return_value='/usr/bin/bwrap'),\
         patch('eesd.recursive_execution.subprocess.run',return_value=SimpleNamespace(stdout='',stderr='bwrap: Cannot mount',returncode=1)):
        with pytest.raises(RuntimeError,match='bubblewrap startup error'):
            run_test('print(3)',{'input':'','expected':'3'})


def test_feedback_requires_distinct_four_tests():
    with pytest.raises(ValueError):feedback_for_tests('print(0)',[{'id':0}]*4)
