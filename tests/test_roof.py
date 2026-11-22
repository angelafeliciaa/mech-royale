"""Offline sanity test for rooftop candidate selection and climb planning (no server needed)."""
import gzip
import json
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from arena_map import ArenaMap, inside  # noqa: E402
from dex_nav import NavGrid  # noqa: E402
from dex_roof import RoofPlanner  # noqa: E402
from dex_tactics import ZoneTracker  # noqa: E402


def load_cfg():
    with gzip.open(os.path.join(ROOT, "kit", "check", "stream.jsonl.gz"), "rt") as f:
        for line in f:
            m = json.loads(line)
            if "config" in m:
                return m["config"]
    raise RuntimeError("no hello in stream")


class RoofTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_cfg()
        cls.arena = ArenaMap.load({"dir": os.path.join(ROOT, "kit", "arena"), "files": ["v2/world.json", "v2/terrain.i16"]})
        cls.nav = NavGrid(cls.arena, cls.cfg["mech"])
        cls.roof = RoofPlanner(cls.arena, cls.nav, cls.cfg["mech"])

    def test_candidates_exist_and_sane(self):
        self.assertGreaterEqual(len(self.roof.cands), 10)
        for c in self.roof.cands:
            self.assertTrue(12.0 <= c.h <= 55.0, c.h)
            self.assertGreaterEqual(c.clear, 8.0)
            cx, cy = c.center
            self.assertTrue(inside(cx, cy, c.outer))
            self.assertAlmostEqual(self.nav.top_at(cx, cy), c.top, delta=0.3)
            self.assertTrue(self.roof.on_roof_ok(c, cx, cy, 3.0))
            self.assertTrue(self.roof.is_on(c, cx, cy, c.top + 0.1))
            self.assertFalse(self.roof.is_on(c, cx, cy, self.arena.floor(cx, cy)))

    def test_climb_plans(self):
        for c in self.roof.cands:
            self.assertTrue(c.launches)
            for (lx, ly), (dx, dy), climb in c.launches:
                self.assertTrue(self.nav.is_free(lx, ly))
                self.assertTrue(self.nav.reachable(lx, ly))
                self.assertTrue(8.0 <= climb <= 55.0)
                self.assertTrue(inside(dx, dy, c.outer))
                self.assertAlmostEqual(self.nav.top_at(dx, dy), c.top, delta=0.3)
                self.assertLess(math.hypot(lx - dx, ly - dy), 12.0)
                # launch stands on open ground (no eave above)
                self.assertLess(self.nav.top_at(lx, ly) - self.arena.floor(lx, ly), 0.6)

    def test_edge_point_stays_on_roof(self):
        c = self.roof.cands[0]
        cx, cy = c.center
        for ang in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
            tx, ty = cx + math.cos(ang) * 150.0, cy + math.sin(ang) * 150.0
            px, py = self.roof.edge_point(c, tx, ty, 3.5)
            self.assertTrue(self.roof.on_roof_ok(c, px, py, 3.4))
            self.assertGreater(math.hypot(px - cx, py - cy), 3.0)

    def test_pick_respects_zone(self):
        zt = ZoneTracker()
        zone = {"center": [41.3, -72.8], "radius": 560.0, "next_center": [41.3, -72.8], "next_radius": 380.0,
                "shrink_start": 60.0, "shrink_end": 110.0, "dps": 20}
        zt.update(zone, 5.0)
        picked = self.roof.pick(0.0, 0.0, zt, zone, 5.0, {})
        self.assertIsNotNone(picked)
        c, (launch, land, climb) = picked
        self.assertLess(math.hypot(c.center[0] - 41.3, c.center[1] + 72.8), 380.0 - 6.0)
        tiny = dict(zone, next_center=[-1000.0, -1000.0], next_radius=40.0)
        self.assertIsNone(self.roof.pick(0.0, 0.0, zt, tiny, 5.0, {}))
        bad = {cc.idx: 1e9 for cc in self.roof.cands}
        self.assertIsNone(self.roof.pick(0.0, 0.0, zt, zone, 5.0, bad))


if __name__ == "__main__":
    unittest.main()
