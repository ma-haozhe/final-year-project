import sys
import time

def progress_bar(value, bar_length=20):
    """Displays a progress bar in the console.

    Arguments:
    value -- a float between 0 and 1 representing the progress
    bar_length -- the length of the progress bar in characters (default 20)
    """
    if value < 0 or value > 1:
        raise ValueError("Value must be between 0 and 1.")

    progress = int(value * bar_length)
    bar = "█" * progress + "░" * (bar_length - progress)
    sys.stdout.write(f"\rProgress: [{bar}] {value * 100:.2f}%")
    sys.stdout.flush()

def delay():
    for i in range (60):
        fl = i/60
        time.sleep(0.2)
        #print(fl)
        progress_bar(fl)

delay()
        