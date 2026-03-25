"""
Process GPX files from chartplotter and/or Inreach MINI.
Output format is determined by the output file extension: .gpx writes GPX, .kml writes KML.

Examples:
python3 gpx_processor.py WaypointsRoutesTracks.gpx out.gpx --start 2024-07-01T00:00:00 --end 2024-12-31T23:59:59 --timezone Europe/Istanbul
python3 gpx_processor.py WaypointsRoutesTracks.gpx out.kml --start 2024-07-01T00:00:00 --end 2024-12-31T23:59:59 --timezone Europe/Istanbul
python3 gpx_processor.py explore_including_belize.gpx belize.gpx --start 2024-12-01T00:00:00 --end 2025-10-01T23:59:59 --timezone America/Belize --join --split
python3 gpx_processor.py WaypointsRoutesTracks3.gpx turkey.gpx --start 2024-08-01T00:00:00 --end 2024-10-01T23:59:59 --timezone Europe/Istanbul --join --split --clean --simplify 4

Workflow for 2024:
# process Inreach download
python3 gpx_processor.py explore.gpx explore_clean.gpx --start 2024-07-01T00:00:00 --end 2024-10-01T23:59:59 --timezone Europe/Istanbul  --join --split --clean --simplify 4
# Process chartplotter download with -3 offset for broken timestamps
python3 gpx_processor.py WaypointsRoutesTracks3.gpx WaypointsRoutesTracks3_clean.gpx --start 2024-07-01T00:00:00 --end 2024-10-01T23:59:59 --timezone Europe/Istanbul --timedelta -3 --join --split --clean --simplify 4
# Merge
python3 gpx_processor.py WaypointsRoutesTracks3_clean.gpx turkey.gpx --backup explore_clean.gpx       

Workflow for 2025:
python3 gpx_processor.py explore2025.gpx explore2025_clean.gpx --start 2025-05-01T00:00:00 --end 2025-10-01T23:59:59 --timezone Europe/Istanbul --join --clean --simplify 5
python3 gpx_processor.py WaypointsRoutesTracks2025.gpx WaypointsRoutesTracks2025_clean.gpx --start 2025-05-01T00:00:00 --end 2025-10-01T23:59:59 --timezone Europe/Istanbul --timedelta -3 --join --clean --simplify 5
python3 gpx_processor.py WaypointsRoutesTracks2025_clean.gpx summer2025_clean.gpx --backup explore2025_clean.gpx --gap 600
# The merge algorithm assumes single segments, so split last
python3 gpx_processor.py summer2025_clean.gpx summer2025.gpx --split --waypoints summer2025_waypoints_annotated.gpx
# Export to KML for Google Earth Pro
python3 gpx_processor.py summer2025.gpx summer2025.kml
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from zoneinfo import ZoneInfo # pip3 install backports-zoneinfo
import dateutil # pip3 install python-dateutil
from pyproj import Geod # pip3 install pyproj
from collections import defaultdict
from typing import List, Dict, Type
import argparse
import functools

# Custom GPX extension namespace for storing track colors
EXT_NS = 'https://gpx-processor/v1'
ET.register_namespace('ext', EXT_NS)

# Named color palettes: hex RGB strings (RRGGBB)
PALETTES: Dict[str, List[str]] = {
    'rainbow': ['FF0000', 'FF7F00', 'FFFF00', '00FF00', '0000FF', '8B00FF'],
    'redgreen': ['d53e4f', 'f46d43', 'fdae61', 'fee08b', 'e6f598', 'abdda4', '66c2a5']
}

def rgb_to_kml(rgb: str) -> str:
    """Convert RRGGBB hex string to KML color format aabbggrr (fully opaque)."""
    r, g, b = rgb[0:2], rgb[2:4], rgb[4:6]
    return f'ff{b}{g}{r}'.lower()

@dataclass
class Waypoint:
    lat: float
    lon: float
    elevation: float = None
    name: float = None
    sog: float = None
    time: datetime = None

    def distance(self, other):
        "Calculate distance between two waypoints in meters, using equirectangular approximation"
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [self.lat, self.lon, other.lat, other.lon])
        lat_mean = (lat1 + lat2) / 2.0
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        R = 6371000  # Radius of the Earth in meters
        x = dlon * math.cos(lat_mean)
        return math.sqrt(x*x + dlat*dlat) * R

    @staticmethod
    def calculate_area(waypoints: List['Waypoint']) -> float:
        "Calculate the area of a list of waypoints using the Shoelace formula"
        assert len(waypoints) >= 3, "Not enough points to form an area"
        if len(waypoints) < 3:
            return 0.0  # Not enough points to form an area

        # Shoelace formula
        area = 0.0
        n = len(waypoints)
        for i in range(n):
            lat1, lon1 = (waypoints[i].lat, waypoints[i].lon)
            lat2, lon2 = (waypoints[(i + 1) % n].lat, waypoints[(i + 1) % n].lon)
            area += lat1 * lon2 - lat2 * lon1

        area = abs(area) / 2.0
        lat_mean = sum([w.lat for w in waypoints]) / n

        # Convert the area from degrees squared to meters squared
        R = 6371000  # Radius of the Earth in meters
        area *= (math.radians(R) ** 2) * math.cos(math.radians(lat_mean))

        return area

    @staticmethod
    def calculate_geodesic_area(waypoints: List['Waypoint']) -> float:
        """Calculate the geodesic area of a polygon in meters² using the WGS84 ellipsoid (via pyproj).

        More accurate than calculate_area for large polygons or high latitudes, at higher
        computational cost. Suitable as a reference for validating calculate_area.
        """
        geod = Geod(ellps='WGS84')
        lons = [w.lon for w in waypoints]
        lats = [w.lat for w in waypoints]
        area, _ = geod.polygon_area_perimeter(lons, lats)
        return abs(area)

    @staticmethod
    def calculate_all_signed_areas(waypoints: List['Waypoint']) -> List[float]:
        """Return a list where areas[i] is the signed shoelace area of waypoints[0..i] (in degrees²).

        Each consecutive cross-product is computed exactly once and accumulated.
        The closing term (last point → first point) is O(1) to update per step.
        Areas[0] and areas[1] are 0.0 (fewer than 3 points cannot form a polygon).
        """
        n = len(waypoints)
        areas = [0.0] * n
        if n < 3:
            return areas

        lat0, lon0 = waypoints[0].lat, waypoints[0].lon
        # Seed with cp_0: cross product of the edge p0→p1
        cumsum = lat0 * waypoints[1].lon - waypoints[1].lat * lon0

        for i in range(2, n):
            lat_prev, lon_prev = waypoints[i - 1].lat, waypoints[i - 1].lon
            lat_i,    lon_i    = waypoints[i].lat,     waypoints[i].lon
            # Accumulate cp_{i-1}: cross product of edge p_{i-1}→p_i
            cumsum += lat_prev * lon_i - lat_i * lon_prev
            # Closing term: edge p_i→p_0
            closing = lat_i * lon0 - lat0 * lon_i
            areas[i] = 0.5 * (cumsum + closing)

        return areas
    

@dataclass
class Route:
    name: str = None
    description: str = None
    points: List[Waypoint] = field(default_factory=list)

@dataclass
class Track:
    name: str = None
    description: str = None
    segments: List[List[Waypoint]] = field(default_factory=list)
    color: str = None  # hex RGB, e.g. "FF0000"; applies to all segments in this track

    def is_within(self, start: datetime = None, end: datetime = None):
        for segment in self.segments:
            for point in segment:
                if point.time:
                    if start and point.time < start:
                        return False
                    if end and point.time > end:
                        return False
        return True
    
    def remove_duplicate_times(self):
        clean_segments = []
        for segment in self.segments:
            seen_times = set()
            clean_segment = []
            for point in segment:
                if point.time not in seen_times:
                    clean_segment.append(point)
                    seen_times.add(point.time)
            print("Cleaned ", len(segment) - len(clean_segment), " duplicate points out of ", len(segment))
            clean_segments.append(clean_segment)
        self.segments = clean_segments

    def simplify(self, threshold: float):
        """Greedily remove maximal sequences of waypoints whose area is less
        than a given threshold"""
        simplified_segments = []
        max_simplify = 1
        for segment in self.segments:
            simplified_segment = [segment[0]]
            try_waypoints = [segment[0], segment[1]]
            for waypoint in segment[2:]:
                try_waypoints.append(waypoint)
                area = Waypoint.calculate_area(try_waypoints)
                if area > threshold:
                    if len(try_waypoints) > max_simplify:
                        max_simplify = len(try_waypoints)
                    # Either of two things are the case here:
                    # 1. We are testing only three waypoints and they fail the
                    # area test. We can do no simplification, so we add the
                    # second waypoint to the simplified segment and continue.
                    # 2. We are testing more than three waypoints and the previous
                    # iteration passed the area test. We add the second-to-last
                    # waypoints to the simplified segment and continue.
                    # Either way, the following is what we want:
                    simplified_segment.append(try_waypoints[-2])
                    try_waypoints = try_waypoints[-2:]
            # Finish the segment with the same logic as above.
            simplified_segment += try_waypoints[-2:]
            print("Simplified ", len(segment) - len(simplified_segment), " duplicate points out of ", len(segment))
            print("Longest removed segment was ", max_simplify, " waypoints")            
            simplified_segments.append(simplified_segment)
        self.segments = simplified_segments

    def prune(self, distance=None, duration=None, count=None):
        for segment in self.segments:
            # Any two points less than prune_distance and prune_duration apart (in space and time)
            # will disqualify all the points in between. So this is our first pass:
            # TODO(martin): Not yet implemented
            pass
            
    def fill_gaps(self, other, gap_threshold: timedelta):
        filled_segments = []
        for (segment_count, segment) in enumerate(self.segments):
            filled_segment = [segment[0]]
            last_waypoint = segment[0]
            for waypoint in segment[1:]:
                if waypoint.time - last_waypoint.time > gap_threshold:
                    # fill from other
                    gap_count = 0
                    for other_segment in other.segments:
                        for other_waypoint in other_segment:
                            if other_waypoint.time > last_waypoint.time and other_waypoint.time < waypoint.time:
                                filled_segment.append(other_waypoint)
                                gap_count += 1
                    print(f"Segment {segment_count}: Gap from {last_waypoint.time} to {waypoint.time} filled with {gap_count} waypoints")
                filled_segment.append(waypoint)
                last_waypoint = waypoint
            filled_segments.append(filled_segment)
        self.segments = filled_segments



@dataclass
class GPX:
    waypoints: List[Waypoint] = field(default_factory=list)
    routes: List[Route] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)




def parse_waypoint(wpt_elem, timezone=None, delta=timedelta()):
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    lat = float(wpt_elem.get('lat'))
    lon = float(wpt_elem.get('lon'))
    ele_elem = wpt_elem.find('gpx:ele', ns)
    elevation = float(ele_elem.text) if ele_elem is not None else None
    name_elem = wpt_elem.find('gpx:name', ns)
    name = name_elem.text if name_elem is not None else None
    sog_elem = wpt_elem.find('gpx:sog', ns)
    sog = float(sog_elem.text) if sog_elem is not None else None
    time_elem = wpt_elem.find('gpx:time', ns)
    time = dateutil.parser.isoparse(time_elem.text) if time_elem is not None else None
    if time and timezone:
        time = time.astimezone(timezone) + delta

    return Waypoint(lat, lon, elevation, name, sog, time)

def read_gpx(filename, timezone=None, delta=timedelta()):
    tree = ET.parse(filename)
    root = tree.getroot()
    gpx = GPX()

    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}

    for wpt in root.findall('gpx:wpt', ns):
        gpx.waypoints.append(parse_waypoint(wpt, timezone=timezone, delta=delta))

    for rte in root.findall('gpx:rte', ns):
        route = Route()
        route.name = rte.find('gpx:name', ns).text if rte.find('gpx:name', ns) is not None else None
        route.description = rte.find('gpx:desc', ns).text if rte.find('gpx:desc', ns) is not None else None
        for rtept in rte.findall('gpx:rtept', ns):
            route.points.append(parse_waypoint(rtept, timezone=timezone, delta=delta))
        gpx.routes.append(route)

    for trk in root.findall('gpx:trk', ns):
        track = Track()
        track.name = trk.find('gpx:name', ns).text if trk.find('gpx:name', ns) is not None else None
        track.description = trk.find('gpx:desc', ns).text if trk.find('gpx:desc', ns) is not None else None
        ext = trk.find('gpx:extensions', ns)
        if ext is not None:
            color_elem = ext.find(f'{{{EXT_NS}}}color')
            if color_elem is not None:
                track.color = color_elem.text
        for trkseg in trk.findall('gpx:trkseg', ns):
            segment = []
            for trkpt in trkseg.findall('gpx:trkpt', ns):
                segment.append(parse_waypoint(trkpt, timezone=timezone, delta=delta))
            track.segments.append(segment)
        gpx.tracks.append(track)

    return gpx

def create_waypoint_elem(wpt, tag='wpt', number=None):
    wpt_elem = ET.Element(tag, {'lat': str(wpt.lat), 'lon': str(wpt.lon)})
    # we don't preserve original numbers, just generate consecutive values
    if number is not None:
        ET.SubElement(wpt_elem, 'number').text = str(number)
    if wpt.elevation is not None:
        ET.SubElement(wpt_elem, 'ele').text = str(wpt.elevation)
    if wpt.name is not None:
        ET.SubElement(wpt_elem, 'name').text = wpt.name
    if wpt.sog is not None:
        ET.SubElement(wpt_elem, 'sog').text = str(wpt.sog)
    if wpt.time is not None:
        ET.SubElement(wpt_elem, 'time').text = wpt.time.isoformat(timespec='seconds')
    #    if wpt.description is not None:
    #        ET.SubElement(wpt_elem, 'desc').text = wpt.description
    return wpt_elem


def write_kml(gpx, filename):
    root = ET.Element('kml', {'xmlns': 'http://www.opengis.net/kml/2.2'})
    doc = ET.SubElement(root, 'Document')

    if gpx.waypoints:
        folder = ET.SubElement(doc, 'Folder')
        ET.SubElement(folder, 'name').text = 'Waypoints'
        for wpt in gpx.waypoints:
            pm = ET.SubElement(folder, 'Placemark')
            if wpt.name:
                ET.SubElement(pm, 'name').text = wpt.name
            point = ET.SubElement(pm, 'Point')
            ele = wpt.elevation if wpt.elevation is not None else 0
            ET.SubElement(point, 'coordinates').text = f'{wpt.lon},{wpt.lat},{ele}'

    if gpx.routes:
        folder = ET.SubElement(doc, 'Folder')
        ET.SubElement(folder, 'name').text = 'Routes'
        for rte in gpx.routes:
            pm = ET.SubElement(folder, 'Placemark')
            if rte.name:
                ET.SubElement(pm, 'name').text = rte.name
            ls = ET.SubElement(pm, 'LineString')
            ET.SubElement(ls, 'tessellate').text = '1'
            ET.SubElement(ls, 'coordinates').text = ' '.join(
                f'{pt.lon},{pt.lat},{pt.elevation if pt.elevation is not None else 0}'
                for pt in rte.points
            )

    if gpx.tracks:
        folder = ET.SubElement(doc, 'Folder')
        ET.SubElement(folder, 'name').text = 'Tracks'
        for trk in gpx.tracks:
            pm = ET.SubElement(folder, 'Placemark')
            if trk.name:
                ET.SubElement(pm, 'name').text = trk.name
            if trk.color:
                style = ET.SubElement(pm, 'Style')
                line_style = ET.SubElement(style, 'LineStyle')
                ET.SubElement(line_style, 'color').text = rgb_to_kml(trk.color)
                ET.SubElement(line_style, 'width').text = '5'
            geom = ET.SubElement(pm, 'MultiGeometry')
            for segment in trk.segments:
                ls = ET.SubElement(geom, 'LineString')
                ET.SubElement(ls, 'tessellate').text = '1'
                ET.SubElement(ls, 'coordinates').text = ' '.join(
                    f'{pt.lon},{pt.lat},{pt.elevation if pt.elevation is not None else 0}'
                    for pt in segment
                )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(filename, encoding='utf-8', xml_declaration=True)


def write_gpx(gpx, filename):
    root = ET.Element('gpx', {
        'version': '1.1',
        'creator': 'Python GPX Writer',
        'xmlns': 'http://www.topografix.com/GPX/1/1'
    })

    for (i_wpt, wpt) in enumerate(gpx.waypoints):
        root.append(create_waypoint_elem(wpt, number=i_wpt))

    for (i_rte, rte) in enumerate(gpx.routes):
        rte_elem = ET.SubElement(root, 'rte')
        # we always generate consecutive values for `number` field
        ET.SubElement(rte_elem, 'number').text = str(i_rte)
        if rte.name is not None:
            ET.SubElement(rte_elem, 'name').text = rte.name
        if rte.description is not None:
            ET.SubElement(rte_elem, 'desc').text = rte.description
        for rtept in rte.points:
            rte_elem.append(create_waypoint_elem(rtept, 'rtept'))

    for (i_trk, trk) in enumerate(gpx.tracks):
        trk_elem = ET.SubElement(root, 'trk')
        # we always generate consecutive values for `number` field
        ET.SubElement(trk_elem, 'number').text = str(i_trk)
        if trk.name is not None:
            ET.SubElement(trk_elem, 'name').text = trk.name
        if trk.description is not None:
            ET.SubElement(trk_elem, 'desc').text = trk.description
        if trk.color is not None:
            ext_elem = ET.SubElement(trk_elem, 'extensions')
            ET.SubElement(ext_elem, f'{{{EXT_NS}}}color').text = trk.color
        for segment in trk.segments:
            trkseg_elem = ET.SubElement(trk_elem, 'trkseg')
            for trkpt in segment:
                trkseg_elem.append(create_waypoint_elem(trkpt, 'trkpt'))

    tree = ET.ElementTree(root)
    # Not pretty:
    # tree.write(filename, encoding='utf-8', xml_declaration=True)
    # Python 3.9 only:
    ET.indent(tree, space="\t", level=0)
    tree.write(filename, encoding='utf-8', xml_declaration=True)
    # xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    #with open(filename, "w") as f:
    #    f.write(xmlstr)

def read_and_process(input_file, timezone, args):
    # Read GPX file
    gpx_data = read_gpx(input_file, timezone=timezone, delta=args.timedelta)

    # Remove routes and waypoints
    gpx_data.waypoints.clear()
    gpx_data.routes.clear()

    # remove anything too early or late
    gpx_data.tracks = [track for track in gpx_data.tracks if track.is_within(args.start, args.end)]

    if args.join:
        combined_track = Track()
        all_segments = sorted([segment for track in gpx_data.tracks for segment in track.segments], key=lambda x: x[0].time)
        combined_segment = sum(all_segments, [])
        combined_track.segments.append(combined_segment)
        gpx_data.tracks = [combined_track]

    for track in gpx_data.tracks:
        if args.clean:
            track.remove_duplicate_times()

        if args.simplify is not None:
            track.simplify(args.simplify)

        # if args.prune:
        #     track.prune(distance=args.prune_distance, duration=args.prune_duration,
        #                 count=args.prune_count)

    if args.split:
        day_segments = defaultdict(list)
        day_tracks = []
        for track in gpx_data.tracks:
            for segment in track.segments:
                for point in segment:
                    day_segments[point.time.date()].append(point)
        for (date, day_segment) in sorted(day_segments.items()):
            day_track = Track()
            day_track.name = str(date)
            day_track.segments.append(day_segment)
            day_tracks.append(day_track)
            if args.split_waypoints:
                # add waypoint at end of each day segment (probably just one segment)
                gpx_data.waypoints.append(Waypoint(
                    lat=day_segment[-1].lat,
                    lon=day_segment[-1].lon,
                    name=day_track.name))
        gpx_data.tracks = day_tracks

    return gpx_data


def main():
    parser = argparse.ArgumentParser(description="Process GPX files")
    parser.add_argument("input_file", help="Input GPX file to process")
    parser.add_argument("output_file", help="Output GPX file to write")
    parser.add_argument("--start", default=None, type=datetime.fromisoformat,
                        help="Earliest date allowed in ISO format")
    parser.add_argument("--end", default=None, type=datetime.fromisoformat,
                        help="Last date allowed in ISO format")
    parser.add_argument("--timezone", default=None, type=str, help="Time zone string")
    parser.add_argument("--timedelta", default=0.0, type=float, help="Time delta in hours")
    parser.add_argument("--join", action="store_true", default=False, help="Combine all tracks")
    parser.add_argument("--split", action="store_true", default=False, help="Split tracks by day")
    parser.add_argument("--split-waypoints", action="store_true", default=False,
                        help="Emit a waypoint for each end of day")
    parser.add_argument("--clean", action="store_true", default=False,
                        help="Remove points with the exact same timestamp as a previous trackpoint")
    parser.add_argument("--simplify", type=float, default=None,
                        help="Remove point sequences with area less than a threshold in square meters")
    # parser.add_argument("--prune", action="store_true", default=False, help="Prune idle time")
    # parser.add_argument("--prune-distance", type=float, default=100,
    #                     help="All points must be within this distance in meters to prune")
    # parser.add_argument("--prune-duration", type=float, default=3600,
    #                     help="All points must be within this time in seconds to prune")
    # parser.add_argument("--prune-count", type=float, default=None,
    #                     help="Must have a least this many points to prune")
    parser.add_argument("--color", nargs='?', const='rainbow', default=None,
                        metavar='PALETTE',
                        help=f"Color each track from a named palette (default: rainbow). "
                             f"Available: {', '.join(PALETTES)}")
    parser.add_argument("--backup", default=None, help="Backup track data to fill gaps")
    parser.add_argument("--waypoints", default=None, help="File with waypoints to add")
    parser.add_argument("--gap", type=float, default=300,
                        help="Minimum gap in seconds to fill from backup")
    args = parser.parse_args()
    tz = ZoneInfo(args.timezone) if args.timezone else timezone.utc
    # note: timedelta is only for the GPX data
    if args.start:
        args.start = args.start.astimezone(tz)
    if args.end:
        args.end = args.end.astimezone(tz)
    args.timedelta = timedelta(hours=args.timedelta)
    
    gpx_data = read_and_process(args.input_file, timezone=tz, args=args)
    if args.backup:
        backup_args = args.copy()
        backup_args.split = False
        backup_data = read_and_process(args.backup, timezone=tz, args=backup_args)
    
        # usually there will be just one of each, so this nested loop is a no-op.
        # indeed this doesn't really work with multiple segments
        for track in gpx_data.tracks:
            for backup_track in backup_data.tracks:
                print("Calling fill_gaps()")
                track.fill_gaps(backup_track, gap_threshold=timedelta(seconds=args.gap))

    if args.waypoints:
        waypoint_data = read_gpx(args.waypoints, timezone=tz, delta=args.timedelta)
        gpx_data.waypoints.extend(waypoint_data.waypoints)

    if args.color:
        palette = PALETTES.get(args.color)
        if palette is None:
            parser.error(f"Unknown palette '{args.color}'. Available: {', '.join(PALETTES)}")
        for i, track in enumerate(gpx_data.tracks):
            track.color = palette[i % len(palette)]

    # Write output — format determined by output file extension
    ext = args.output_file.rsplit('.', 1)[-1].lower()
    if ext == 'kml':
        write_kml(gpx_data, args.output_file)
    else:
        write_gpx(gpx_data, args.output_file)

    print(f"Processed {args.input_file} and wrote results to {args.output_file}")

if __name__ == "__main__":
    main()
