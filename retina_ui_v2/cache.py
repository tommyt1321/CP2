"""Private per-process upload cache. Adapted from the local v1 adapter."""
import atexit
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

def lock_file(handle):
    handle.seek(0)
    if sys.platform == 'win32':
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


class Cache:
    """Only clean app-owned directories whose OS lifetime lock is released."""
    MARKER = 'RETINA_INSIGHT_LOCAL_CACHE_V2'
    def __init__(self, base=None):
        self.base = Path(base or Path(tempfile.gettempdir())/'retina_insight_ui_v2')
        if self.base.is_symlink(): raise ValueError('Cache root cannot be a symlink.')
        self.base.mkdir(parents=True, exist_ok=True, mode=0o700)
        for p in self.base.glob('run_*'):
            if p.is_symlink() or not p.is_dir(): continue
            marker, lock = p/'owner.txt', p/'lifetime.lock'
            try:
                if marker.is_symlink() or lock.is_symlink() or marker.read_text() != self.MARKER: continue
                with lock.open('r+b') as h: lock_file(h)
                shutil.rmtree(p)
            except (OSError, UnicodeError):
                pass  # Active process, unrecognised folder, or Windows open handle.
        self.path = self.base/('run_'+uuid.uuid4().hex)
        self.path.mkdir(mode=0o700)
        (self.path/'owner.txt').write_text(self.MARKER)
        lock = self.path/'lifetime.lock'; lock.write_bytes(b'0')
        self.handle = lock.open('r+b'); lock_file(self.handle)
        self.closed = False
        atexit.register(self.close)

    def close(self):
        if self.closed: return
        self.closed = True; self.handle.close()
        try: shutil.rmtree(self.path)
        except OSError: pass  # Retry as stale owned directory on a later startup.


