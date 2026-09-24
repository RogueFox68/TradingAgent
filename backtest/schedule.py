"""What each arm's bots could see in active_targets.json, and from when.

A TargetSchedule answers the only question the bots ask of the scout: "at
time t, what does active_targets.json hold?" - reproducing the fleet's own
loader semantics (utils.load_and_validate_targets):

  * the newest PUBLISHED file wins; a run that failed or never transferred
    changes nothing on the Beelink;
  * a file whose `updated` stamp is more than 24h old is ignored and every
    bucket reads EMPTY (no static fallback: fleet_bot passes {}). This is why
    Monday mornings trade nothing until the 08:30 CT run lands - Friday's
    15:00 CT file is ~65h old;
  * `updated` is stamped when sector_scout_3 STARTS (run_scout builds the
    dict before analysing anything), so file age runs from Phase 2 launch,
    not from publication.

The arms differ only in which candidates each published run contributes,
what confidence they carry, and when they become visible.
"""
import bisect
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

from . import rules
from .scout_log import EQUITY_BUCKETS

ET = ZoneInfo("America/New_York")
BUCKETS = EQUITY_BUCKETS + ("wheel_targets",)   # wheel feeds survivor's blacklist
EMPTY = {b: {} for b in BUCKETS}


@dataclass
class Snapshot:
    effective: datetime      # first instant the Beelink can read it
    updated: datetime        # the file's `updated` stamp (drives the 24h rule)
    targets: dict            # bucket -> {symbol: confidence}
    run_id: int = -1


class TargetSchedule:
    def __init__(self, name, snapshots, max_age_s=rules.TARGETS_MAX_AGE_S,
                 survivor_blacklist=True, both_directions=False):
        self.name = name
        self.snapshots = sorted(snapshots, key=lambda s: s.effective)
        self._keys = [s.effective for s in self.snapshots]
        self.max_age_s = max_age_s
        # survivor_bot skips any symbol on the trend or wheel list. Meaningless
        # when every bucket is the same list (top-N), where it would blank
        # survivor entirely, so that arm turns it off.
        self.survivor_blacklist = survivor_blacklist
        # trend_bot tries LONG for a symbol on trend_targets and only reaches
        # SHORT via `elif` for one that is not. With one shared list (top-N)
        # that would forbid shorts, so that arm lets both directions fire.
        self.both_directions = both_directions

    def at(self, t):
        i = bisect.bisect_right(self._keys, t) - 1
        if i < 0:
            return EMPTY
        snap = self.snapshots[i]
        if (t - snap.updated).total_seconds() > self.max_age_s:
            return EMPTY
        return snap.targets

    def symbols(self):
        out = set()
        for s in self.snapshots:
            for b in EQUITY_BUCKETS:
                out.update(s.targets.get(b, {}))
        return out

    def run_at(self, t):
        i = bisect.bisect_right(self._keys, t) - 1
        return self.snapshots[i].run_id if i >= 0 else -1


def published_runs(runs, start, end):
    """Published runs whose publication falls in [start - 4 days, end]. The
    lead-in lets a window that opens on a Monday see Friday's file (which the
    24h rule then correctly ignores until Monday's first run lands)."""
    lo = start - timedelta(days=4)
    return [r for r in runs if r.published and lo <= r.scout_done <= end]


def flat_confidence(runs):
    """Median confidence of APPROVED equity-bucket candidates. The no-LLM
    arms size every trade at this value, so their average position size
    matches the LLM arm's and the comparison isolates selection from sizing."""
    vals = [c.confidence for r in runs for c in r.candidates
            if c.approved and c.bucket in EQUITY_BUCKETS]
    return round(statistics.median(vals), 4) if vals else 0.75


def _snap(run, i, pick, conf, at):
    targets = {b: {} for b in BUCKETS}
    for c in run.candidates:
        if c.bucket in targets and pick(c):
            targets[c.bucket][c.symbol] = c.confidence if conf is None else conf
    if at == "publish":
        eff, upd = run.scout_done, run.scout_started or run.scout_done
    elif at == "scanner":
        # A pipeline with no LLM publishes when the scanner finishes, and
        # would stamp `updated` then.
        eff = upd = run.scanner_done or run.started
    else:
        raise ValueError(at)
    return Snapshot(eff, upd, targets, i)


def llm_schedule(runs, flat_conf=None, at="publish", name=None):
    """What the fleet actually received: approved candidates only."""
    snaps = [_snap(r, i, lambda c: c.approved, flat_conf, at) for i, r in enumerate(runs)]
    return TargetSchedule(name or "llm", snaps)


def all_candidates_schedule(runs, flat_conf, at, name=None):
    """The scanner's output with the LLM removed: every candidate approved."""
    snaps = [_snap(r, i, lambda c: True, flat_conf, at) for i, r in enumerate(runs)]
    return TargetSchedule(name or f"all_{at}", snaps)


def random_schedule(runs, flat_conf, seed, name=None):
    """The null for "the LLM only helps because it trades less": per run and
    per bucket, approve the SAME NUMBER of candidates the LLM approved, chosen
    uniformly at random. Same count, same timing, same sizing - only the
    choice differs."""
    rng = random.Random(seed)
    snaps = []
    for i, r in enumerate(runs):
        chosen = set()
        for b in BUCKETS:
            cands = r.bucket(b)
            k = sum(c.approved for c in cands)
            chosen.update((b, c.symbol) for c in rng.sample(cands, k))
        snaps.append(_snap(r, i, lambda c, ch=chosen: (c.bucket, c.symbol) in ch,
                           flat_conf, "publish"))
    return TargetSchedule(name or f"random_{seed}", snaps)


def topn_schedule(daily_lists, flat_conf, name="topN"):
    """Arm A: the point-in-time top-N list, available from the open of the
    day it was ranked for (it is built from the PRIOR sessions' volume)."""
    snaps = []
    for i, (day, syms) in enumerate(sorted(daily_lists.items())):
        eff = datetime.combine(day, dtime(9, 0), tzinfo=ET)
        t = {s: flat_conf for s in syms}
        targets = {"trend_targets": dict(t), "short_targets": dict(t),
                   "survivor_targets": dict(t), "wheel_targets": {}}
        snaps.append(Snapshot(eff, eff, targets, i))
    return TargetSchedule(name, snaps, survivor_blacklist=False, both_directions=True)
