"""Check account-reported usage before each optimizer model call.

A missing or unreadable requested window fails closed; a weekly window must
never be interpreted as a daily window. Uses only account/rateLimits/read.
"""
import json
import select
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


class QuotaUnavailable(SystemExit):
    pass


class QuotaReached(SystemExit):
    pass


def check_snapshot(snapshot, window_minutes, threshold=13):
    limits = snapshot.get('rateLimitsByLimitId') or {}
    codex = limits.get('codex') or snapshot.get('rateLimits') or {}
    matching = [x for x in (codex.get('primary'),codex.get('secondary'))
                if x and x.get('windowDurationMins') == window_minutes]
    if len(matching) != 1 or not isinstance(matching[0].get('usedPercent'),(int,float)):
        raise QuotaUnavailable(f'account does not report an unambiguous {window_minutes}-minute Codex quota')
    used = float(matching[0]['usedPercent'])
    if used >= threshold:
        raise QuotaReached(f'quota usage {used:g}% reached {threshold:g}% for the {window_minutes}-minute window')
    return used


def read_snapshot(timeout_s=15):
    proc = subprocess.Popen(['codex','app-server'], stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,text=True,bufsize=1)
    def send(msg):
        proc.stdin.write(json.dumps(msg)+'\n')
        proc.stdin.flush()
    def response(expected):
        deadline=time.monotonic()+timeout_s
        while time.monotonic()<deadline:
            ready,_,_=select.select([proc.stdout],[],[],deadline-time.monotonic())
            if not ready:
                break
            line=proc.stdout.readline()
            if not line:
                break
            try:
                obj=json.loads(line)
            except ValueError:
                continue
            if obj.get('id')==expected:
                if obj.get('error'):
                    raise QuotaUnavailable('account rate-limit API returned an error')
                return obj.get('result') or {}
        raise QuotaUnavailable('account rate-limit API timed out or stopped')
    try:
        send({'id':1,'method':'initialize','params':{'clientInfo':{'name':'autorag-quota-guard','version':'1.0'}}})
        response(1)
        send({'method':'initialized','params':{}})
        send({'id':2,'method':'account/rateLimits/read','params':{}})
        return response(2)
    except (OSError, BrokenPipeError) as exc:
        raise QuotaUnavailable(f'account quota check failed: {type(exc).__name__}') from exc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def enforce(policy_path, marker_path):
    policy_path, marker_path = Path(policy_path), Path(marker_path)
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {'window_minutes':1440,'threshold_percent':13}
    try:
        return check_snapshot(read_snapshot(), policy['window_minutes'], policy['threshold_percent'])
    except (QuotaUnavailable, QuotaReached) as exc:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker = {'timestamp':datetime.now(timezone.utc).isoformat(), 'policy':policy,
                  'condition':'threshold_reached' if isinstance(exc,QuotaReached) else 'quota_unavailable',
                  'reason':str(exc)}
        tmp = marker_path.with_name(marker_path.name+'.tmp')
        tmp.write_text(json.dumps(marker, indent=2)+'\n')
        tmp.replace(marker_path)
        raise
