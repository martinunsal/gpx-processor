"""
Unit tests for gpx_processor.py — GPX read/write roundtrip tests.

Run with:
    python3 -m pytest test_gpx_processor.py -v
or:
    python3 -m unittest test_gpx_processor -v
"""

import tempfile
import os
import math
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from gpx_processor import GPX, Waypoint, Route, Track, read_gpx, write_gpx, write_kml, PALETTES, EXT_NS, rgb_to_kml


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


class TestCalculateAllSignedAreas(unittest.TestCase):

    def _make_square(self, side=1.0):
        """A square with vertices at (0,0),(side,0),(side,side),(0,side). Area = side²."""
        return [
            Waypoint(lat=0.0,    lon=0.0),
            Waypoint(lat=side,   lon=0.0),
            Waypoint(lat=side,   lon=side),
            Waypoint(lat=0.0,    lon=side),
        ]

    def test_short_lists(self):
        """0, 1, and 2 points all return all-zero lists."""
        for n in (0, 1, 2):
            pts = [Waypoint(float(i), float(i)) for i in range(n)]
            areas = Waypoint.calculate_all_signed_areas(pts)
            self.assertEqual(len(areas), n)
            self.assertTrue(all(a == 0.0 for a in areas))

    def test_triangle_area(self):
        """Right triangle with legs 1.0 has area 0.5."""
        pts = [Waypoint(0.0, 0.0), Waypoint(1.0, 0.0), Waypoint(0.0, 1.0)]
        areas = Waypoint.calculate_all_signed_areas(pts)
        self.assertAlmostEqual(abs(areas[2]), 0.5)

    def test_square_area(self):
        """Unit square has area 1.0 at index 3."""
        pts = self._make_square(side=1.0)
        areas = Waypoint.calculate_all_signed_areas(pts)
        self.assertEqual(len(areas), 4)
        self.assertEqual(areas[0], 0.0)
        self.assertEqual(areas[1], 0.0)
        self.assertAlmostEqual(abs(areas[3]), 1.0)

    def test_prefix_areas_are_independent(self):
        """areas[i] reflects only the first i+1 points, not later ones."""
        pts = self._make_square(side=2.0)
        areas = Waypoint.calculate_all_signed_areas(pts)
        # Triangle (first 3 points of a 2×2 square): area = 2.0
        self.assertAlmostEqual(abs(areas[2]), 2.0)
        # Full square: area = 4.0
        self.assertAlmostEqual(abs(areas[3]), 4.0)

    def test_matches_calculate_area_for_all_prefixes(self):
        """calculate_all_signed_areas |areas[i]| matches calculate_area on each prefix.

        calculate_area scales to meters²; calculate_all_signed_areas returns degrees².
        We verify the ratio is constant (cos(lat_mean) * (R * π/180)²) across all prefixes.
        """
        pts = [
            Waypoint(lat=36.94588261, lon=28.09343972),
            Waypoint(lat=36.94588981, lon=28.09331353),
            Waypoint(lat=36.94588261, lon=28.0932955),
            Waypoint(lat=36.92415369, lon=28.1621126),
            Waypoint(lat=36.9241681,  lon=28.1621126),
        ]
        all_areas = Waypoint.calculate_all_signed_areas(pts)

        R = 6371000
        scale = (math.radians(R) ** 2)

        for i in range(3, len(pts)):
            prefix = pts[:i + 1]
            expected_m2 = Waypoint.calculate_area(prefix)
            lat_mean = sum(w.lat for w in prefix) / len(prefix)
            # Convert degrees² → meters² using the same scaling as calculate_area
            computed_m2 = abs(all_areas[i]) * scale * math.cos(math.radians(lat_mean))
            self.assertAlmostEqual(computed_m2, expected_m2, places=6)

    def test_collinear_points_zero_area(self):
        """Collinear points produce zero area."""
        pts = [Waypoint(float(i), float(i)) for i in range(5)]
        areas = Waypoint.calculate_all_signed_areas(pts)
        for a in areas:
            self.assertAlmostEqual(a, 0.0)


class TestCalculateArea(unittest.TestCase):
    """Tests for calculate_area using calculate_geodesic_area as the ground truth."""

    def _rect(self, lat0, lat1, lon0, lon1):
        """Rectangle vertices in counterclockwise order."""
        return [
            Waypoint(lat=lat0, lon=lon0),
            Waypoint(lat=lat0, lon=lon1),
            Waypoint(lat=lat1, lon=lon1),
            Waypoint(lat=lat1, lon=lon0),
        ]

    def _check_vs_geodesic(self, waypoints, tol=0.005):
        approx = Waypoint.calculate_area(waypoints)
        geodesic = Waypoint.calculate_geodesic_area(waypoints)
        rel_err = abs(approx - geodesic) / geodesic
        self.assertLess(rel_err, tol,
                        f"Relative error {rel_err:.4%} exceeds {tol:.1%}: "
                        f"approx={approx:.2f} geodesic={geodesic:.2f}")

    def test_small_triangle_vs_geodesic(self):
        """Tiny real-world triangle near Marmaris: approx area within 1% of geodesic."""
        waypoints = [
            Waypoint(lat=36.94588261, lon=28.09343972),
            Waypoint(lat=36.94588981, lon=28.09331353),
            Waypoint(lat=36.94588261, lon=28.0932955),
        ]
        area = Waypoint.calculate_area(waypoints)
        geodesic = Waypoint.calculate_geodesic_area(waypoints)
        self.assertLess(abs(area - geodesic) / geodesic, 0.01)

    def test_collinear_meridian_zero_area(self):
        """Points sharing the same longitude (on a meridian) have zero area."""
        waypoints = [
            Waypoint(lat=36.92415369, lon=28.1621126),
            Waypoint(lat=36.9241681,  lon=28.1621126),
            Waypoint(lat=36.92424736, lon=28.1621126),
            Waypoint(lat=36.9242978,  lon=28.1621126),
            Waypoint(lat=36.92430501, lon=28.1621126),
        ]
        self.assertEqual(Waypoint.calculate_area(waypoints), 0.0)

    def test_small_rect_equator(self):
        """0.1°×0.1° square at the equator: within 0.5% of geodesic."""
        self._check_vs_geodesic(self._rect(-0.05, 0.05, 0.0, 0.1))

    def test_small_rect_60N(self):
        """0.1°×0.1° square at 60°N: within 1% of geodesic (sphere vs WGS84 at high lat)."""
        self._check_vs_geodesic(self._rect(59.95, 60.05, 0.0, 0.1), tol=0.01)

    def test_area_ratio_equator_vs_60N(self):
        """Area at 60°N should be ~cos(60°)=0.5× the equatorial area for same angular size.

        Both our spherical approximation and the WGS84 geodesic should be within a few
        percent of cos(60°). We don't compare them to each other because they use different
        Earth models (sphere vs ellipsoid).
        """
        pts_eq  = self._rect(-0.05, 0.05, 0.0, 0.1)
        pts_60n = self._rect(59.95, 60.05, 0.0, 0.1)
        cos60 = math.cos(math.radians(60.0))  # 0.5
        ratio_approx   = Waypoint.calculate_area(pts_60n)   / Waypoint.calculate_area(pts_eq)
        ratio_geodesic = Waypoint.calculate_geodesic_area(pts_60n) / Waypoint.calculate_geodesic_area(pts_eq)
        # Spherical formula: exact cos(lat) scaling, so ratio should be very close
        self.assertAlmostEqual(ratio_approx, cos60, delta=0.001)
        # WGS84 geodesic: within 2% (ellipsoid vs sphere difference at 60°N)
        self.assertAlmostEqual(ratio_geodesic, cos60, delta=0.02)

    def test_matches_calculate_area_for_all_prefixes(self):
        """calculate_all_signed_areas scaled to m² matches calculate_area on each prefix."""
        pts = [
            Waypoint(lat=36.94588261, lon=28.09343972),
            Waypoint(lat=36.94588981, lon=28.09331353),
            Waypoint(lat=36.94588261, lon=28.0932955),
            Waypoint(lat=36.92415369, lon=28.1621126),
            Waypoint(lat=36.9241681,  lon=28.1621126),
        ]
        all_areas = Waypoint.calculate_all_signed_areas(pts)

        R = 6371000
        scale = (math.radians(R) ** 2)

        for i in range(3, len(pts)):
            prefix = pts[:i + 1]
            expected_m2 = Waypoint.calculate_area(prefix)
            lat_mean = sum(w.lat for w in prefix) / len(prefix)
            computed_m2 = abs(all_areas[i]) * scale * math.cos(math.radians(lat_mean))
            self.assertAlmostEqual(computed_m2, expected_m2, places=6)


class TestTrackColor(unittest.TestCase):

    def test_color_roundtrip_gpx(self):
        """Track color survives a GPX write/read cycle."""
        t = datetime(2024, 8, 1, 8, 0, 0, tzinfo=timezone.utc)
        trk = Track(name='Red Day', segments=[[Waypoint(1.0, 2.0, time=t)]], color='FF0000')
        result = roundtrip(GPX(tracks=[trk]))
        self.assertEqual(result.tracks[0].color, 'FF0000')

    def test_no_color_roundtrip_gpx(self):
        """Track without color reads back with color=None."""
        t = datetime(2024, 8, 1, 8, 0, 0, tzinfo=timezone.utc)
        trk = Track(name='Uncolored', segments=[[Waypoint(1.0, 2.0, time=t)]])
        result = roundtrip(GPX(tracks=[trk]))
        self.assertIsNone(result.tracks[0].color)

    def test_multiple_track_colors_roundtrip(self):
        """Multiple tracks each retain their own color."""
        t = datetime(2024, 8, 1, 8, 0, 0, tzinfo=timezone.utc)
        colors = PALETTES['rainbow']
        tracks = [Track(name=f'Day {i}', segments=[[Waypoint(float(i), float(i), time=t)]], color=colors[i])
                  for i in range(len(colors))]
        result = roundtrip(GPX(tracks=tracks))
        for i, trk in enumerate(result.tracks):
            self.assertEqual(trk.color, colors[i])

    def test_rgb_to_kml_conversion(self):
        """RRGGBB → aabbggrr with full opacity."""
        self.assertEqual(rgb_to_kml('FF0000'), 'ff0000ff')  # red
        self.assertEqual(rgb_to_kml('00FF00'), 'ff00ff00')  # green
        self.assertEqual(rgb_to_kml('0000FF'), 'ffff0000')  # blue
        self.assertEqual(rgb_to_kml('FF7F00'), 'ff007fff')  # orange


class TestWriteKML(unittest.TestCase):

    def _write_and_parse(self, gpx):
        """Write gpx to a KML temp file and return the parsed XML root."""
        f = tempfile.NamedTemporaryFile(suffix='.kml', delete=False)
        f.close()
        try:
            write_kml(gpx, f.name)
            return ET.parse(f.name).getroot()
        finally:
            os.unlink(f.name)

    def test_kml_track_placemark(self):
        """Each track produces a Placemark with a LineString inside MultiGeometry."""
        t = datetime(2024, 8, 1, 8, 0, 0, tzinfo=timezone.utc)
        trk = Track(name='Day 1', segments=[[Waypoint(36.9, 28.1, elevation=0.0, time=t)]])
        root = self._write_and_parse(GPX(tracks=[trk]))
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        placemarks = root.findall('.//kml:Placemark', ns)
        self.assertEqual(len(placemarks), 1)
        self.assertEqual(placemarks[0].find('kml:name', ns).text, 'Day 1')
        self.assertIsNotNone(placemarks[0].find('.//kml:LineString', ns))

    def test_kml_track_color_style(self):
        """Colored track gets a Style/LineStyle element with the right KML color."""
        t = datetime(2024, 8, 1, 8, 0, 0, tzinfo=timezone.utc)
        trk = Track(name='Red', segments=[[Waypoint(1.0, 2.0, time=t)]], color='FF0000')
        root = self._write_and_parse(GPX(tracks=[trk]))
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        color_elem = root.find('.//kml:LineStyle/kml:color', ns)
        self.assertIsNotNone(color_elem)
        self.assertEqual(color_elem.text, 'ff0000ff')

    def test_kml_no_color_no_style(self):
        """Track without color has no Style element."""
        t = datetime(2024, 8, 1, 8, 0, 0, tzinfo=timezone.utc)
        trk = Track(name='Plain', segments=[[Waypoint(1.0, 2.0, time=t)]])
        root = self._write_and_parse(GPX(tracks=[trk]))
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        self.assertIsNone(root.find('.//kml:Style', ns))

    def test_kml_waypoint_placemark(self):
        """Waypoints produce Point Placemarks with correct coordinates (lon,lat,ele)."""
        wpt = Waypoint(lat=36.9, lon=28.1, elevation=5.0, name='Harbour')
        root = self._write_and_parse(GPX(waypoints=[wpt]))
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        coords = root.find('.//kml:Point/kml:coordinates', ns).text
        self.assertEqual(coords, '28.1,36.9,5.0')


if __name__ == '__main__':
    unittest.main()
