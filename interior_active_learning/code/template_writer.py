"""Deferred, coalescing writer for paint/<image>_paint_template.png.

WHY. A Whole-region click on a 25 MP frame cost 1.49 s end to end, measured on
MAR_Amb_AS_ETD_0003: 0.58 s to render the overlay and 0.93 s to PNG-encode and write
22.8 MB. The reviewer waits on all of it, but only the render affects the response -- the
client is handed a cropped patch of the changed rectangle (9 ms), and the correction mask,
the only source of truth, was already written synchronously under mask_lock before this
point. The 0.93 s encode is bookkeeping for the NEXT reader, so it does not belong on the
request.

WHAT THIS IS NOT. It is not a write-back cache for label data. Every correction is still
written to disk synchronously on the request thread; nothing a reviewer records can be lost
here. The template is a *render* of that mask and is regenerable at any time
(regenerate_templates.py rebuilds all of them).

STALENESS, and why it is bounded. Every in-process reader calls flush() first, so no
endpoint can observe a template older than the last edit -- that is the whole contract, and
test_app.py checks it. Two further backstops: writes are atomic (temp file + os.replace), so
unlike the previous plain Image.save() a concurrent reader can never decode a
half-written PNG; and if the process is killed with a write still pending, /api/template
already detects a correction mask newer than the template and re-renders from the cached
stage. A crash therefore costs a re-render, not a wrong overlay.

The one case not covered is an out-of-process reader (build_figures.py, or the workflow
diagram script) started in the same second as a click. That races with the old synchronous
save too; the atomic replace makes it strictly safer than before.
"""

import atexit
import os
import threading
import time

_PENDING = {}            # path -> PIL.Image, latest wins (a newer render supersedes)
_INFLIGHT = set()        # paths currently being encoded
_COND = threading.Condition()
_THREAD = None
_FAILURES = []           # (path, exception) for the last few failed writes
_FAILED_PATHS = {}       # path -> exception, cleared when that path is written successfully


def _write_atomic(path, img):
    tmp = f"{path}.tmp{os.getpid()}"
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def _worker():
    while True:
        with _COND:
            while not _PENDING:
                _COND.wait()
            path = next(iter(_PENDING))
            img = _PENDING.pop(path)
            _INFLIGHT.add(path)
        try:
            _write_atomic(path, img)
        except Exception as exc:                      # noqa: BLE001
            # A failed template write must not take the server down, and must not be
            # silent either. Recorded per-path so flush() can return False for it: the
            # first version only appended to a list nothing in production reads, then
            # cleared _INFLIGHT in the finally, so flush() found nothing outstanding and
            # answered True for a file that was never written. A reader was told "drained"
            # and went on to read a stale overlay -- the failure mode the barrier exists
            # to prevent, reintroduced by the error path.
            _FAILURES.append((path, exc))
            del _FAILURES[:-5]
            with _COND:
                _FAILED_PATHS[path] = exc
            print(f"template write failed for {os.path.basename(path)}: "
                  f"{type(exc).__name__}: {exc}")
        else:
            with _COND:
                _FAILED_PATHS.pop(path, None)
        finally:
            with _COND:
                _INFLIGHT.discard(path)
                _COND.notify_all()


def queue(path, img):
    """Schedule `img` to become the contents of `path`. Returns immediately."""
    global _THREAD
    with _COND:
        _PENDING[path] = img
        if _THREAD is None or not _THREAD.is_alive():
            _THREAD = threading.Thread(target=_worker, name="template-writer",
                                       daemon=True)
            _THREAD.start()
        _COND.notify_all()


def flush(path=None, timeout=120.0):
    """Block until pending writes (for `path`, or all of them) are on disk.

    Call this before reading a template file. Returns True only if the queue drained
    AND the last write for that path succeeded. False means the file you are about to
    read may be behind -- either the queue timed out, or the write raised.
    """
    deadline = time.monotonic() + timeout
    with _COND:
        while True:
            outstanding = [p for p in list(_PENDING) + list(_INFLIGHT)
                           if path is None or p == path]
            if not outstanding:
                failed = [p for p in _FAILED_PATHS if path is None or p == path]
                return not failed
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _COND.wait(min(remaining, 0.5))


def path_for_read(path):
    """flush(path) then hand the path back, so a read site states its own barrier.

    Written as a wrapper rather than a bare flush() call at the top of each function
    because the barrier belongs to the READ, not to the endpoint: someone adding a new
    reader copies `open(path_for_read(p))` and gets it right, where they could easily
    copy `open(p)` and silently reintroduce the stale-read bug.
    """
    if not flush(path):
        # Do not pretend this is current. The correction mask is authoritative and
        # unaffected; the overlay may be one edit behind, and saying so beats a reader
        # silently rendering stale pixels.
        print(f"WARNING: {os.path.basename(path)} may be stale -- a deferred overlay "
              f"write did not complete. Re-open the image to re-render it.")
    return path


def discard(path):
    """Drop any queued write for `path` because the caller is about to write it itself.

    A WRITE site needs this, not flush(). app_undo and hybrid_detect both render and save
    the template synchronously; if a deferred write for the same path were still queued it
    would land AFTERWARDS and overwrite their newer render with an older one -- an undo
    that silently un-did itself a second later. Waits out an in-flight encode for the same
    path so the caller's os.replace is unambiguously last.
    """
    with _COND:
        _PENDING.pop(path, None)
        while path in _INFLIGHT:
            _COND.wait(0.5)


def pending_count():
    with _COND:
        return len(_PENDING) + len(_INFLIGHT)


def failures():
    return list(_FAILURES)


# A pending template on exit is a re-render for whoever opens the image next, which is
# recoverable but pointless to inflict. Give the queue a chance to drain.
atexit.register(lambda: flush(timeout=60.0))
