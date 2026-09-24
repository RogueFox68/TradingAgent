"""Parse scout_log.txt into the record of what the LLM actually decided.

scout_log.txt is the only archive of the scout's decisions: run_scout.bat
APPENDS every run to it, and sector_scout_3.py prints every candidate it
analysed - rejected ones included - with its technical and per-source LLM
sub-scores. active_targets.json is overwritten each run, and the LLM cannot be
re-run retroactively (yfinance only serves current headlines), so this file is
the ground truth for every LLM arm of the backtest.

The approval emoji is taken as authoritative rather than re-deriving approval
from `Conf >= threshold`: the threshold and the weighting both changed over
the life of the log (early runs approved at ~0.50), and a re-derivation would
silently apply today's rule to last spring's decisions.

Timestamps come from run_scout.bat's `[%DATE% %TIME%]` stamps, which are the
Corsair's local clock (Central Time). Windows pads single-digit hours with a
space: `[Thu 09/24/2026  8:30:00.11]`.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

LOG_TZ = "America/Chicago"

# Buckets the equity bots consume. condor_targets appears in early runs and
# wheel_targets feeds wheel_bot (options, out of scope for the first pass);
# both are parsed and kept, but the simulator only reads the equity buckets.
EQUITY_BUCKETS = ("trend_targets", "short_targets", "survivor_targets")

_STAMP = re.compile(
    r"^\[(?:\w{3}) (\d{2})/(\d{2})/(\d{4}) +(\d{1,2}):(\d{2}):(\d{2})\.(\d{2})\] ?(.*)$")
_BUCKET = re.compile(r"👉 Analyzing (\w+)\.\.\.")
_CANDIDATE = re.compile(
    r"^\s+(✅|❌) (\S+)\s*\| Conf:\s*([\d.]+) \["
    r"Tech: ([\d.]+|N/A) \| T1: ([\d.]+|N/A) \| T2: ([\d.]+|N/A) \| "
    r"T3: ([\d.]+|N/A) \| Soc: ([\d.]+|N/A)\]")
_AI_ERROR = re.compile(r"\[!\] AI Error on (\S+?):")
# Newer scouts score a failed LLM call as missing: it prints "N/A" like a
# source with no coverage and is named after the breakdown instead.
_LLM_FAILED = re.compile(r"\(LLM failed: ([\w, ]+)\)")


def _num(tok):
    return None if tok == "N/A" else float(tok)


@dataclass
class Candidate:
    bucket: str
    symbol: str
    approved: bool
    confidence: float
    tech: float            # normalised tech score, 0-1
    t1: float | None       # elite news
    t2: float | None       # mainstream news
    t3: float | None       # specialty news
    social: float | None
    ai_error: bool = False
    # Sources whose LLM call failed, from the "(LLM failed: T1, Soc)" suffix.
    # Their score reads None (N/A), but the source HAD coverage. Older logs
    # print a failed call as 0.00 instead and leave this empty.
    failed: tuple = ()

    def has_news(self):
        """Any news in any tier, whether or not its LLM call succeeded."""
        return (any(x is not None for x in (self.t1, self.t2, self.t3))
                or any(s in self.failed for s in ("T1", "T2", "T3")))

    def llm_score(self):
        """The LLM's share of the composite under the CURRENT weights
        (T1 .30, T2 .20, T3 .10, Social .10; N/A counts as the neutral 0.5 the
        scout substitutes), renormalised to 0-1. This is the part of the
        confidence the technical score does not explain."""
        parts = ((self.t1, 0.30), (self.t2, 0.20), (self.t3, 0.10), (self.social, 0.10))
        total = sum(w * (0.5 if s is None else s) for s, w in parts)
        return total / 0.70


@dataclass
class Run:
    started: datetime                  # run_scout.bat start (tz-aware)
    scanner_done: datetime | None = None
    scanner_ok: bool = True
    scout_started: datetime | None = None   # Phase 2 launch; `updated` in the file
    scout_done: datetime | None = None       # "Scout Complete" - targets published
    scout_ok: bool = False
    transfer_ok: bool = False
    candidates: list = field(default_factory=list)

    @property
    def published(self):
        """True if the Beelink received this run's targets."""
        return self.scout_ok and self.transfer_ok and self.scout_done is not None

    @property
    def llm_latency_s(self):
        if self.scanner_done is None or self.scout_done is None:
            return None
        return (self.scout_done - self.scanner_done).total_seconds()

    def bucket(self, name):
        return [c for c in self.candidates if c.bucket == name]


def _stamp(line, tz):
    m = _STAMP.match(line)
    if not m:
        return None, None
    mo, d, y, hh, mm, ss, cs = (int(g) for g in m.groups()[:7])
    dt = datetime(y, mo, d, hh, mm, ss, cs * 10000, tzinfo=tz)
    return dt, m.group(8)


def parse(lines, tz_name=LOG_TZ):
    """Parse an iterable of log lines into a list of Runs, oldest first.

    A run is opened by run_scout.bat's STARTING line. Candidate lines outside
    a run (there should be none) are ignored rather than guessed at.
    """
    tz = ZoneInfo(tz_name)
    runs = []
    run = None
    bucket = None
    ai_errors = set()
    for raw in lines:
        line = raw.rstrip("\r\n")
        dt, rest = _stamp(line, tz)
        if dt is not None:
            if "STARTING DAILY TRADING SEQUENCE" in rest:
                run = Run(started=dt)
                runs.append(run)
                bucket, ai_errors = None, set()
            elif run is None:
                continue
            elif "Scanner Complete" in rest:
                run.scanner_done, run.scanner_ok = dt, True
            elif "Scanner Failed" in rest:
                run.scanner_done, run.scanner_ok = dt, False
            elif "Launching Sector Scout" in rest:
                run.scout_started = dt
            elif "Scout Complete" in rest:
                run.scout_done, run.scout_ok = dt, True
            elif "Scout failed" in rest:
                run.scout_done, run.scout_ok = dt, False
            continue
        if run is None:
            continue
        if "Transfer Complete" in line:
            run.transfer_ok = True
            continue
        m = _BUCKET.search(line)
        if m:
            bucket = m.group(1)
            continue
        m = _AI_ERROR.search(line)
        if m:
            ai_errors.add(m.group(1))
            continue
        m = _CANDIDATE.match(line)
        if m and bucket:
            emoji, sym, conf, tech, t1, t2, t3, soc = m.groups()
            f = _LLM_FAILED.search(line, m.end())
            run.candidates.append(Candidate(
                bucket=bucket, symbol=sym, approved=(emoji == "✅"),
                confidence=float(conf), tech=_num(tech) or 0.0,
                t1=_num(t1), t2=_num(t2), t3=_num(t3), social=_num(soc),
                ai_error=sym in ai_errors,
                failed=tuple(x.strip() for x in f.group(1).split(",")) if f else ()))
    return runs


def load(path, tz_name=LOG_TZ):
    with open(path, encoding="utf-8", errors="replace") as f:
        return parse(f, tz_name)
