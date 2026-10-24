import os


def prevent_idle_sleep() -> None:
    """Keep Windows from idle-sleeping while this process runs.

    Uses SetThreadExecutionState, which only lasts as long as the calling
    thread/process; no power settings are changed. No-op on other platforms.
    """
    if os.name != "nt":
        return
    import ctypes

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
