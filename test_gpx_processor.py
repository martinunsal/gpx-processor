"""
Unit tests for gpx_processor.py — GPX read/write roundtrip tests.

Run with:
    python3 -m pytest test_gpx_processor.py -v
or:
    python3 -m unittest test_gpx_processor -v
"""

import tempfile
import os
import unittest
from datetime import datetime, timezone

from gpx_processor import GPX, Waypoint, Route, Track, read_gpx, write_gpx


def make_temp_gpx(content: str) -> str:
    """Write content to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.gpx', delete=False, encoding='utf-8')
    f.write(content)
    f.close()
    return f.name


def roundtrip(gpx: GPX) -> GPX:
    """Write gpx to a temp file, read it back, return the result."""
    f = tempfile.NamedTemporaryFile(suffix='.gpx', delete=False)
    f.close()
    try:
        write_gpx(gpx, f.name)
        return read_gpx(f.name)
    finally:
        os.unlink(f.name)


class TestReadGPX(unittest.TestCase):

    def test_read_waypoint(self):
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <wpt lat="36.9459" lon="28.0934">
    <ele>5.0</ele>
    <name>Harbour</name>
    <time>2024-08-01T10:00:00+00:00</time>
  </wpt>
</gpx>"""
        path = make_temp_gpx(xml)
        try:
            gpx = read_gpx(path)
        finally:
            os.unlink(path)

        self.assertEqual(len(gpx.waypoints), 1)
        wpt = gpx.waypoints[0]
        self.assertAlmostEqual(wpt.lat, 36.9459)
        self.assertAlmostEqual(wpt.lon, 28.0934)
        self.assertAlmostEqual(wpt.elevation, 5.0)
        self.assertEqual(wpt.name, 'Harbour')
        self.assertEqual(wpt.time, datetime(2024, 8, 1, 10, 0, 0, tzinfo=timezone.utc))

    def test_read_track(self):
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <name>Day 1</name>
    <trkseg>
      <trkpt lat="36.9000" lon="28.1000"><ele>0.0</ele><time>2024-08-01T08:00:00+00:00</time></trkpt>
      <trkpt lat="36.9100" lon="28.1100"><ele>0.0</ele><time>2024-08-01T09:00:00+00:00</time></trkpt>
    </trkseg>
  </trk>
</gpx>"""
        path = make_temp_gpx(xml)
        try:
            gpx = read_gpx(path)
        finally:
            os.unlink(path)

        self.assertEqual(len(gpx.tracks), 1)
        trk = gpx.tracks[0]
        self.assertEqual(trk.name, 'Day 1')
        self.assertEqual(len(trk.segments), 1)
        self.assertEqual(len(trk.segments[0]), 2)
        self.assertAlmostEqual(trk.segments[0][0].lat, 36.9000)
        self.assertAlmostEqual(trk.segments[0][1].lon, 28.1100)

    def test_read_multiple_segments(self):
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <name>Multi</name>
    <trkseg>
      <trkpt lat="1.0" lon="1.0"/>
    </trkseg>
    <trkseg>
      <trkpt lat="2.0" lon="2.0"/>
      <trkpt lat="3.0" lon="3.0"/>
    </trkseg>
  </trk>
</gpx>"""
        path = make_temp_gpx(xml)
        try:
            gpx = read_gpx(path)
        finally:
            os.unlink(path)

        trk = gpx.tracks[0]
        self.assertEqual(len(trk.segments), 2)
        self.assertEqual(len(trk.segments[0]), 1)
        self.assertEqual(len(trk.segments[1]), 2)

    def test_read_optional_fields_absent(self):
        """Waypoint with only lat/lon — all optional fields should be None."""
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <wpt lat="10.0" lon="20.0"/>
</gpx>"""
        path = make_temp_gpx(xml)
        try:
            gpx = read_gpx(path)
        finally:
            os.unlink(path)

        wpt = gpx.waypoints[0]
        self.assertIsNone(wpt.elevation)
        self.assertIsNone(wpt.name)
        self.assertIsNone(wpt.time)


class TestWriteGPX(unittest.TestCase):

    def test_write_waypoint_roundtrip(self):
        t = datetime(2024, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
        gpx = GPX(waypoints=[Waypoint(lat=36.9459, lon=28.0934, elevation=5.0, name='Harbour', time=t)])
        result = roundtrip(gpx)

        self.assertEqual(len(result.waypoints), 1)
        wpt = result.waypoints[0]
        self.assertAlmostEqual(wpt.lat, 36.9459)
        self.assertAlmostEqual(wpt.lon, 28.0934)
        self.assertAlmostEqual(wpt.elevation, 5.0)
        self.assertEqual(wpt.name, 'Harbour')
        self.assertEqual(wpt.time, t)

    def test_write_track_roundtrip(self):
        t1 = datetime(2024, 8, 1, 8, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 8, 1, 9, 0, 0, tzinfo=timezone.utc)
        seg = [Waypoint(36.9, 28.1, elevation=0.0, time=t1),
               Waypoint(36.91, 28.11, elevation=0.0, time=t2)]
        gpx = GPX(tracks=[Track(name='Day 1', segments=[seg])])
        result = roundtrip(gpx)

        self.assertEqual(len(result.tracks), 1)
        trk = result.tracks[0]
        self.assertEqual(trk.name, 'Day 1')
        self.assertEqual(len(trk.segments[0]), 2)
        self.assertAlmostEqual(trk.segments[0][1].lat, 36.91)

    def test_write_multiple_tracks_roundtrip(self):
        t = datetime(2024, 8, 1, 8, 0, 0, tzinfo=timezone.utc)
        trk1 = Track(name='Track A', segments=[[Waypoint(1.0, 2.0, time=t)]])
        trk2 = Track(name='Track B', segments=[[Waypoint(3.0, 4.0, time=t)]])
        gpx = GPX(tracks=[trk1, trk2])
        result = roundtrip(gpx)

        self.assertEqual(len(result.tracks), 2)
        self.assertEqual(result.tracks[0].name, 'Track A')
        self.assertEqual(result.tracks[1].name, 'Track B')

    def test_write_empty_gpx(self):
        gpx = GPX()
        result = roundtrip(gpx)
        self.assertEqual(result.waypoints, [])
        self.assertEqual(result.tracks, [])
        self.assertEqual(result.routes, [])

    def test_waypoint_without_optional_fields(self):
        """Waypoints with only lat/lon should roundtrip without error."""
        gpx = GPX(waypoints=[Waypoint(lat=10.0, lon=20.0)])
        result = roundtrip(gpx)

        wpt = result.waypoints[0]
        self.assertAlmostEqual(wpt.lat, 10.0)
        self.assertAlmostEqual(wpt.lon, 20.0)
        self.assertIsNone(wpt.elevation)
        self.assertIsNone(wpt.name)
        self.assertIsNone(wpt.time)


if __name__ == '__main__':
    unittest.main()
