"""A script that never finishes, for testing the workhorse budget timeout.

Committed rather than written by the test at run time: a test that creates
and deletes a file inside the repo leaves it behind when the run is
interrupted, and tools/workhorse.py refuses to start on a dirty tree.
That happened once, and the stray file then broke the next run's cleanup.
"""
import time

while True:
    time.sleep(3600)
