import time
import threading
import sys
from functools import wraps

def with_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        stop_event = threading.Event()
        result = None

        def timer():
            start_time = time.time()
            while not stop_event.is_set():
                elapsed = time.time() - start_time
                sys.stdout.write(f"\r{elapsed:.1f}s")
                sys.stdout.flush()
                time.sleep(0.1)
            elapsed = time.time()-start_time
            sys.stdout.write(f"\rDone in {elapsed:.1f}s\n")
            sys.stdout.flush()

        thread = threading.Thread(target=timer)
        thread.start()

        try:
            result = func(*args, **kwargs)
        finally:
            stop_event.set()
            thread.join()
        return result
    return wrapper