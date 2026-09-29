import os
import threading

_listeners = []
_listeners_lock = threading.Lock()


def add_listener(listener):
    """Register a callable(message: str, clear_screen: bool) that receives every log message."""
    with _listeners_lock:
        if listener not in _listeners:
            _listeners.append(listener)


def remove_listener(listener):
    with _listeners_lock:
        if listener in _listeners:
            _listeners.remove(listener)


def log(message, clear_screen=False):
    with _listeners_lock:
        listeners = list(_listeners)

    if not listeners:
        if clear_screen:
            os.system('cls' if os.name == 'nt' else 'clear')
        print(message)
        return

    for listener in listeners:
        try:
            listener(str(message), clear_screen)
        except Exception:
            # A misbehaving listener must never break Boxxy's work.
            pass
