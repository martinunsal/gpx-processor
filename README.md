# gpx-processor
My quick and dirty script for processing (splitting, merging, simplifying) GPX files

I'd say the best part of this project is the algorithm for simplifying a GPS track by removing redundant points. In short, it greedily removes successive points as long as the area of the polygon formed by those points (using Shoelace formula) is smaller than a fixed parameter. I probably wasn't the first to invent this but it's definitely better than most of the sketchy heuristics I see for this purpose.

The worst part of this project is that I use equirectangular projection. Stay near the equator and it'll work just fine. :)
