# gpx-processor
My quick and dirty script for processing (splitting, merging, simplifying) GPX files

I'd say the most interesting part of this project is the algorithm for simplifying a GPS track by removing redundant points. In short, it greedily removes successive points as long as the area of the polygon formed by those points (using Shoelace formula) is smaller than a fixed parameter. I probably wasn't the first to invent this but it's definitely better than most ad hoc algorithms I see for this purpose.
