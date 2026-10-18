"""vtl ("video timeline"): a minimal clip finder.

index: per-frame captions + transcript -> an LLM-built timeline of strict events (one versioned file per stage).
find:  brief + timeline -> LLM candidate clips -> LLM decision pass -> word-safe edges -> clips JSON.
Every stage is one small function with a versioned output so it can later be swapped for a distilled local model.
"""
