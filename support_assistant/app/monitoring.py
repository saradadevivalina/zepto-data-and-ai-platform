# monitoring.py
# Lightweight production observability for the /ask endpoint.
#
# Every request is:
#   1. appended as one JSON line to logs/requests.jsonl (timestamp, intent, sources,
#      confidence, latency, error) -> a persistent audit trail that can be analysed later
#   2. added to in-memory counters -> served live by GET /metrics
#
# Monitoring must never break the API, so logging errors are swallowed.

import json
import threading
from collections import Counter, deque
from datetime import datetime, timezone

from app.config import BASE_DIR, MOCK_LLM

LOG_PATH = BASE_DIR / "logs" / "requests.jsonl"
LOW_CONFIDENCE_THRESHOLD = 0.6   # answers below this are counted as "low confidence"
WINDOW = 1000                    # latency / confidence stats use the last 1000 requests

_lock = threading.Lock()
_started_at = datetime.now(timezone.utc)
_total = 0
_errors = 0
_low_confidence = 0
_intents = Counter()             # policy_question vs general_question
_top_sources = Counter()         # which document was retrieved first, e.g. doc_07
_latencies_ms = deque(maxlen=WINDOW)
_confidences = deque(maxlen=WINDOW)


def record_request(query, intent, sources, confidence, latency_ms, error=None):
    """Record one /ask request in the counters and the JSONL log."""
    global _total, _errors, _low_confidence
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query_length": len(query),
        "intent": intent,
        "top_source": sources[0] if sources else None,
        "n_sources": len(sources),
        "confidence": confidence,
        "latency_ms": round(latency_ms, 1),
        "mock_llm": MOCK_LLM,
        "error": error,
    }
    with _lock:
        _total += 1
        _latencies_ms.append(latency_ms)
        if error:
            _errors += 1
        else:
            _intents[intent] += 1
            _confidences.append(confidence)
            if sources:
                _top_sources[sources[0]] += 1
            if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
                _low_confidence += 1
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # never fail a customer request because logging failed


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def get_metrics():
    """Snapshot of service health and usage, returned by GET /metrics."""
    with _lock:
        latencies = list(_latencies_ms)
        confidences = list(_confidences)
        return {
            "total_requests": _total,
            "errors": _errors,
            "error_rate": round(_errors / _total, 4) if _total else 0.0,
            "intent_counts": dict(_intents),
            "top_source_counts": dict(_top_sources.most_common()),
            "latency_ms": {
                "p50": round(_percentile(latencies, 50), 1) if latencies else None,
                "p95": round(_percentile(latencies, 95), 1) if latencies else None,
                "max": round(max(latencies), 1) if latencies else None,
            },
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
            "low_confidence_answers": _low_confidence,
            "mock_llm": MOCK_LLM,
            "uptime_seconds": int((datetime.now(timezone.utc) - _started_at).total_seconds()),
        }
