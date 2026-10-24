"""Optional Langfuse tracing for the AlphaPROBE mining loop.

One search iteration is one trace (an ``agent`` observation named
``alphaprobe-iteration``); all iterations of a run share one session id so the
Sessions view shows the whole run in order. Inside an iteration:

* ``retrieve-parents`` (retriever) -- the nodes picked by the Bayesian retriever;
* ``evolve-node`` (chain), one per selected node, containing
  ``generate-offspring`` (generation: prompts, raw output, model, token usage,
  reasoning) and ``evaluate-offspring`` (evaluator: IC/ICIR/MutIC and outcome
  of every candidate);
* ``evaluate-pool`` (evaluator) on logging steps: pool test IC and the
  adaptive-combination validation/test metrics.

Iteration numbers (insert rate, parse failures, pool size, test metrics) are
attached as trace scores.

Tracing is off unless the ``langfuse`` package is installed and
LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set (LANGFUSE_TRACING=off turns
it off explicitly). Exporter errors are printed and swallowed so tracing never
changes or stops a run.
"""

import os
from contextlib import ExitStack, contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional


try:
    from opentelemetry.sdk.trace.id_generator import IdGenerator as _IdGenerator
except ImportError:
    _IdGenerator = object


class _OsRandomIdGenerator(_IdGenerator):
    """Trace/span ids from OS randomness.

    The default OpenTelemetry generator uses the global ``random`` module, which
    reseed_everything() seeds, so every run with the same seed would reuse the
    same trace and span ids and overwrite earlier runs in Langfuse.
    """

    def generate_span_id(self) -> int:
        return int.from_bytes(os.urandom(8), "big") or 1

    def generate_trace_id(self) -> int:
        return int.from_bytes(os.urandom(16), "big") or 1


class _NullObservation:
    def update(self, **_: Any) -> None:
        return None


class _Observation:
    def __init__(self, remote: Any) -> None:
        self._remote = remote

    def update(self, **fields: Any) -> None:
        try:
            self._remote.update(**{k: v for k, v in fields.items() if v is not None})
        except Exception as exc:
            print(f"[langfuse] update failed: {exc}")


class RunTracer:
    def __init__(self, session_id: str, tags: Optional[List[str]] = None,
                 metadata: Optional[Mapping[str, Any]] = None) -> None:
        self.session_id = session_id[:200]
        self.tags = sorted({t[:200] for t in (tags or []) if t})
        # propagate_attributes only accepts string values of at most 200 characters.
        self.metadata = {str(k): str(v)[:190] for k, v in dict(metadata or {}).items()}
        self._trace_id: Optional[str] = None
        self._client = self._connect()

    @staticmethod
    def _connect() -> Any:
        mode = os.getenv("LANGFUSE_TRACING", "auto").lower()
        if mode == "off":
            return None
        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            if mode == "on":
                print("[langfuse] LANGFUSE_TRACING=on but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are not set; tracing disabled")
            return None
        try:
            from langfuse import Langfuse
        except ImportError:
            print("[langfuse] credentials found but the langfuse package is not installed; tracing disabled")
            return None
        try:
            base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
            return Langfuse(base_url=base_url, id_generator=_OsRandomIdGenerator())
        except Exception as exc:
            print(f"[langfuse] client initialisation failed; tracing disabled: {exc}")
            return None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def iteration(self, step: int, input: Any = None) -> Iterator[Any]:
        """Root observation of one search iteration (one trace)."""
        with ExitStack() as stack:
            if self._client is not None:
                try:
                    from langfuse import propagate_attributes
                    stack.enter_context(propagate_attributes(
                        session_id=self.session_id,
                        trace_name="alphaprobe-iteration",
                        tags=self.tags,
                        metadata={**self.metadata, "step": str(step)},
                    ))
                except Exception as exc:
                    print(f"[langfuse] propagate_attributes failed: {exc}")
            with self.observe("alphaprobe-iteration", as_type="agent", input=input) as root:
                if self._client is not None:
                    try:
                        self._trace_id = self._client.get_current_trace_id()
                    except Exception:
                        self._trace_id = None
                try:
                    yield root
                finally:
                    self._trace_id = None

    @contextmanager
    def observe(self, name: str, as_type: str = "span", input: Any = None,
                metadata: Optional[Mapping[str, Any]] = None, **generation_fields: Any) -> Iterator[Any]:
        """Child observation of whatever observation is current."""
        if self._client is None:
            yield _NullObservation()
            return
        remote_cm = None
        try:
            remote_cm = self._client.start_as_current_observation(
                name=name, as_type=as_type, input=input,
                metadata=dict(metadata) if metadata else None,
                **{k: v for k, v in generation_fields.items() if v is not None},
            )
        except Exception as exc:
            print(f"[langfuse] start observation {name} failed: {exc}")
        with ExitStack() as stack:
            observation: Any = _NullObservation()
            if remote_cm is not None:
                try:
                    observation = _Observation(stack.enter_context(remote_cm))
                except Exception as exc:
                    print(f"[langfuse] enter observation {name} failed: {exc}")
            try:
                yield observation
            except BaseException as exc:
                observation.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
                raise

    def generation(self, name: str, model: str, system_prompt: str, user_prompt: str,
                   model_parameters: Optional[Dict[str, Any]] = None,
                   metadata: Optional[Mapping[str, Any]] = None):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self.observe(name, as_type="generation", input=messages, metadata=metadata,
                            model=model, model_parameters=model_parameters)

    def score(self, name: str, value: float, comment: Optional[str] = None) -> None:
        """Numeric score on the current iteration's trace."""
        if self._client is None or self._trace_id is None or value is None:
            return
        try:
            self._client.create_score(name=name, value=float(value), trace_id=self._trace_id,
                                      data_type="NUMERIC", comment=comment)
        except Exception as exc:
            print(f"[langfuse] score {name} failed: {exc}")

    def flush(self) -> None:
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception as exc:
            print(f"[langfuse] flush failed: {exc}")
