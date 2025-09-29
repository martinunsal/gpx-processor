"""
Process GPX files from chartplotter and/or Inreach MINI

Examples:
python3 gpx_processor.py WaypointsRoutesTracks.gpx out.gpx --start 2024-07-01T00:00:00 --end 2024-12-31T23:59:59 --timezone Europe/Istanbul
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
python3 gpx_processor.py summer2025_clean.gpx summer2025.gpx --split
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from zoneinfo import ZoneInfo # pip3 install backports-zoneinfo 
import dateutil # pip3 install python-dateutil
from collections import defaultdict
from typing import List, Dict, Type
import argparse
import functools

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
        area *= (math.radians(R) ** 2) * math.cos(lat_mean)

        return area


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
        max_simplify = 10
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
    if timezone:
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
        for trkseg in trk.findall('gpx:trkseg', ns):
            segment = []
            for trkpt in trkseg.findall('gpx:trkpt', ns):
                segment.append(parse_waypoint(trkpt, timezone=timezone, delta=delta))
            track.segments.append(segment)
        gpx.tracks.append(track)

    return gpx

def create_waypoint_elem(wpt, tag='wpt'):
    wpt_elem = ET.Element(tag, {'lat': str(wpt.lat), 'lon': str(wpt.lon)})
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


def write_gpx(gpx, filename):
    root = ET.Element('gpx', {
        'version': '1.1',
        'creator': 'Python GPX Writer',
        'xmlns': 'http://www.topografix.com/GPX/1/1'
    })

    for wpt in gpx.waypoints:
        root.append(create_waypoint_elem(wpt))

    for rte in gpx.routes:
        rte_elem = ET.SubElement(root, 'rte')
        if rte.name is not None:
            ET.SubElement(rte_elem, 'name').text = rte.name
        if rte.description is not None:
            ET.SubElement(rte_elem, 'desc').text = rte.description
        for rtept in rte.points:
            rte_elem.append(create_waypoint_elem(rtept, 'rtept'))

    for trk in gpx.tracks:
        trk_elem = ET.SubElement(root, 'trk')
        if trk.name is not None:
            ET.SubElement(trk_elem, 'name').text = trk.name
        if trk.description is not None:
            ET.SubElement(trk_elem, 'desc').text = trk.description
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

def run_tests():
    waypoints = [
        Waypoint(lat=36.94588261, lon=28.09343972),
        Waypoint(lat=36.94588981, lon=28.09331353),
        Waypoint(lat=36.94588261, lon=28.0932955)
    ]
    area = Waypoint.calculate_area(waypoints)
    assert area - 4.682014129243959 < 1e-10

    waypoints = [
        Waypoint(lat=36.92415369, lon=28.1621126),
        Waypoint(lat=36.9241681, lon=28.1621126),
        Waypoint(lat=36.92424736, lon=28.1621126),
        Waypoint(lat=36.9242978, lon=28.1621126),
        Waypoint(lat=36.92430501, lon=28.1621126)
    ]
    area = Waypoint.calculate_area(waypoints)
    assert area == 0.0

def read_and_process(input_file, start, end, timezone, delta, join, split, clean, simplify):
    # Read GPX file
    gpx_data = read_gpx(input_file, timezone=timezone, delta=delta)

    # Remove routes and waypoints
    gpx_data.waypoints.clear()
    gpx_data.routes.clear()

    # remove anything too early or late
    gpx_data.tracks = [track for track in gpx_data.tracks if track.is_within(start, end)]

    if join:
        combined_track = Track()
        all_segments = sorted([segment for track in gpx_data.tracks for segment in track.segments], key=lambda x: x[0].time)
        combined_segment = sum(all_segments, [])
        combined_track.segments.append(combined_segment)
        gpx_data.tracks = [combined_track]

    for track in gpx_data.tracks:
        if clean:
            track.remove_duplicate_times()

        if simplify is not None:
            track.simplify(simplify)

    if split:
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
        gpx_data.tracks = day_tracks

    return gpx_data


def main():
    run_tests()

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
    parser.add_argument("--clean", action="store_true", default=False,
                        help="Remove points with the exact same timestamp as a previous trackpoint")
    parser.add_argument("--simplify", type=float, default=None,
                        help="Remove point sequences with area less than a threshold in square meters")
    parser.add_argument("--prune", action="store_true", default=False, help="Prune idle time")
    parser.add_argument("--prune-radius", type=float, default=100,
                        help="All points must be within this radius in meters to prune")
    parser.add_argument("--prune-duration", type=float, default=3600,
                        help="All points must be within this time in seconds to prune")
    parser.add_argument("--prune-count", type=float, default=None,
                        help="Must have a least this many points to prune")
    parser.add_argument("--backup", default=None, help="Backup track data to fill gaps")
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
    
    gpx_data = read_and_process(args.input_file, start=args.start, end=args.end, timezone=tz,
                                delta=args.timedelta, join=args.join, split=args.split,
                                clean=args.clean, simplify=args.simplify)
    if args.backup:
        backup_data = read_and_process(args.backup, start=args.start, end=args.end,
                                       timezone=tz, delta=args.timedelta, join=args.join,
                                       split=False, clean=args.clean, simplify=args.simplify)
    
        # usually there will be just one of each, so this nested loop is a no-op
        for track in gpx_data.tracks:
            for backup_track in backup_data.tracks:
                print("Calling fill_gaps()")
                track.fill_gaps(backup_track, gap_threshold=timedelta(seconds=args.gap))

    # Write modified data to a new GPX file
    write_gpx(gpx_data, args.output_file)

    print(f"Processed {args.input_file} and wrote results to {args.output_file}")

if __name__ == "__main__":
    main()
