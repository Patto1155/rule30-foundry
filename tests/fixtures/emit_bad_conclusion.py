"""Emits a result whose conclusion is an unqualified 'never'.

Fixture for the regression that postflight must see a script's own
conclusions rather than letting them sit unread under stdout_json.
"""
import json

print(json.dumps({
    "horizon": 1000,
    "metrics": {},
    "conclusions": ["The center column never repeats."],
}))
