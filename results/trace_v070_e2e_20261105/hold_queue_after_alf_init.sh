#!/bin/bash
# User 2026-11-05: exploration q=1 + SemFilter v0.7.1 from the exploration rounds on. The running queue would prepare the
# arm with q=4, so stop it right after the ALF init ends (before readiness / arm prepare). Exact anchored match on new lines.
cd /localhome/hza214/tridb
LOG=results/trace_v070_e2e_20261105/queue.log
START=$(wc -l < $LOG)
until tail -n +$((START + 1)) $LOG | grep -Eq '^[0-9T:+-]+ alfworld init ended with [a-z_]+$'; do
  systemctl --user is-active -q trace-v070-e2e-20261105 || { echo "$(date -Is) hold: queue ended before the ALF init line" >> $LOG; exit 1; }
  sleep 5
done
systemctl --user stop trace-v070-e2e-20261105
echo "$(date -Is) queue HELD after ALF init (user: q=1 and SemFilter v0.7.1 from the exploration rounds); arm not prepared" >> $LOG
