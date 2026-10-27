"""Fail-closed Python stdin/stdout execution with bubblewrap network isolation.

No direct unsandboxed execution of generated programs is provided. The caller
must provision working bubblewrap user namespaces (or a separately audited
sandbox backend). A sandbox failure is fatal, NOT a failed coding test.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import resource
import shutil
import subprocess


@dataclass(frozen=True)
class Outcome:
    status: str
    actual: str
    expected: str
    stderr: str


def _limits(timeout: int, mem_mb: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU,(timeout+1,timeout+1))
    resource.setrlimit(resource.RLIMIT_AS,(mem_mb*1048576,mem_mb*1048576))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))


def run_test(code: str, test: dict, *, timeout: int=2, memory_mb: int=1024) -> Outcome:
    if not isinstance(code,str) or not code.strip() or len(code)>16000:
        return Outcome('COMPILE_ERROR','','','empty or oversized source')
    if not isinstance(test,dict) or not all(k in test for k in ('input','expected')):
        raise ValueError('test must contain input and expected')
    if timeout<1 or memory_mb<128:
        raise ValueError('invalid sandbox limits')
    executable=shutil.which('bwrap')
    if not executable:
        raise RuntimeError('bubblewrap required for untrusted code; fail closed')
    # Expose only standard system Python/lib paths read-only; no workspace mount.
    cmd=[executable,'--die-with-parent','--new-session','--unshare-all',
         '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--chdir','/tmp',
         '--setenv','HOME','/tmp','--setenv','PYTHONNOUSERSITE','1']
    for host in ('/usr','/lib','/lib64','/bin'):
        if Path(host).exists():
            cmd+=['--ro-bind',host,host]
    cmd+=['--','/usr/bin/python3','-I','-B','-c',code]
    try:
        result=subprocess.run(cmd,input=str(test['input']),stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,text=True,timeout=timeout+1,
                    env={'PATH':'/usr/bin:/bin','PYTHONHASHSEED':'0'},
                    preexec_fn=lambda:_limits(timeout,memory_mb))
    except subprocess.TimeoutExpired:
        return Outcome('TIMEOUT','',str(test['expected']),'time limit exceeded')
    except (OSError,subprocess.SubprocessError) as exc:
        raise RuntimeError(f'sandbox infrastructure failed: {exc}') from exc
    if 'bwrap:' in result.stderr or 'bwrap: ' in result.stdout:
        raise RuntimeError(f'bubblewrap startup error: {result.stderr[-500:]}')
    if result.returncode!=0:
        status='COMPILE_ERROR' if 'SyntaxError' in result.stderr else 'RUNTIME_EXCEPTION'
    else:
        status='PASS' if result.stdout.rstrip('\r\n')==str(test['expected']).rstrip('\r\n') else 'WRONG_OUTPUT'
    return Outcome(status,result.stdout[:4096],str(test['expected'])[:4096],result.stderr[-1000:])


def feedback_for_tests(code: str, tests: list[dict]) -> list[dict]:
    if len(tests)!=4 or len({t['id'] for t in tests})!=4:
        raise ValueError('exactly four distinct tests required')
    out=[]
    for t in tests:
        v=run_test(code,t)
        out.append({'id':t['id'],'input':t['input'][:512],
                    'expected':t['expected'][:512],
                    'actual':v.actual[:512],'stderr':v.stderr[:512],
                    'outcome':v.status})
    return out
