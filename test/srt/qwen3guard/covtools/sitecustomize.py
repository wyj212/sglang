"""Line tracer for the changed-line coverage numbers.

The image used for this port ships no ``coverage`` package, so this installs a
bare ``sys.settrace`` tracer instead.  Python imports ``sitecustomize`` at
startup, so putting this directory on ``PYTHONPATH`` and setting
``QGUARD_COV=1`` is enough to instrument a process - and, because the variable
is inherited, any scheduler subprocess it spawns too.

Every process writes its own dump keyed by pid, so a multiprocess run unions
cleanly instead of racing on one file.
"""

import atexit
import os
import sys
import threading

ROOT = os.environ.get("QGUARD_COV_ROOT", "")
OUT = os.environ.get("QGUARD_COV_OUT", "/tmp/qguard_cov")


def _install(root: str) -> None:
    rows: set[tuple[str, int]] = set()

    def local(frame, event, arg):
        if event == "line":
            rows.add((frame.f_code.co_filename, frame.f_lineno))
        return local

    def global_tracer(frame, event, arg):
        # Only descend into the tree we care about; everything else costs
        # nothing after this first check.
        if frame.f_code.co_filename.startswith(root):
            return local
        return None

    def dump() -> None:
        try:
            os.makedirs(OUT, exist_ok=True)
            path = os.path.join(OUT, "cov-%d-%d.data" % (os.getpid(), len(rows)))
            with open(path, "w", encoding="utf-8") as fh:
                for filename, lineno in sorted(rows):
                    fh.write("%s\t%d\n" % (filename, lineno))
        except OSError:
            pass

    sys.settrace(global_tracer)
    threading.settrace(global_tracer)
    atexit.register(dump)


if os.environ.get("QGUARD_COV") == "1":
    if not ROOT:
        sys.stderr.write("QGUARD_COV=1 needs QGUARD_COV_ROOT\n")
    else:
        _install(ROOT)
