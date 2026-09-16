# ResQTwin Geospatial Data & Network Record

## GCC Ward Boundary
- Wards: 200
- CRS: EPSG:4326
- Invalid geometries: 0

## GCC Road Centreline
- Original features: 37,347
- Exploded segments: 56,649
- CRS: EPSG:4326
- Invalid geometries: 0
- Unique road IDs: 36,942
- Missing road IDs: 110
- Duplicated road IDs: 286
- Exact duplicate geometries: 2

## Road Network
- Noded edges: 111,090
- Nodes: 88,541
- Total network length: 7,016.96 km
- Average edge length: 63.16 m
- Maximum edge length: 2,694.95 m
- Isolated nodes: 0
- Node mapping failures: 0
- Self-loops: 24

## Current Issues
- Investigate 24 self-loops
- Investigate extremely short edges
- Test connected components
- Validate graph before routing

## Milestone 5 — Road Graph Construction & Quality Validation

| Item                       |            Value |
| -------------------------- | ---------------: |
| Original road features     |       **37,347** |
| Exploded road segments     |       **56,649** |
| Noded road edges           |      **111,090** |
| Road nodes                 |       **88,541** |
| Total network length       |  **7,016.96 km** |
| Average edge length        |      **63.16 m** |
| Median edge length         |      **44.61 m** |
| Shortest edge              |   **0.000002 m** |
| Longest edge               |   **2,694.95 m** |
| Connected components       |           **94** |
| Largest component          | **87,752 nodes** |
| Largest component coverage |       **99.11%** |
| Isolated nodes             |            **0** |
| Self-loops                 |           **24** |
| Edges ≤ 1 m                |        **2,194** |
| Edges ≤ 5 m                |        **6,473** |
| Edges ≤ 10 m               |       **10,014** |
| Node mapping failures      |            **0** |


## Milestone — Road Graph Construction & Quality Validation

| Metric                       |            Value |
| ---------------------------- | ---------------: |
| Original road features       |       **37,347** |
| Exploded road segments       |       **56,649** |
| Final graph nodes            |       **88,541** |
| Final graph edges            |      **111,090** |
| CRS                          |    **EPSG:4326** |
| Total network length         |  **7,016.96 km** |
| Average edge length          |      **63.16 m** |
| Median edge length           |      **44.61 m** |
| Maximum edge length          |   **2,694.95 m** |
| Self-loop edges              |           **24** |
| Parallel node pairs          |          **183** |
| Extra parallel edges         |          **184** |
| Connected components         |           **94** |
| Largest component            | **87,752 nodes** |
| Largest component coverage   |       **99.11%** |
| Small components (≤10 nodes) |           **77** |
| Isolated nodes               |            **0** |
| Maximum node degree          |            **5** |
| Degree-1 nodes               |       **21,085** |
| Degree-2 nodes               |        **4,539** |
| Degree ≥3 nodes              |       **62,917** |
| Edges ≤1 m                   |        **2,194** |
| Edges ≤5 m                   |        **6,473** |
| Edges ≤10 m                  |       **10,014** |
| Invalid node references      |            **0** |
| Zero-length edges            |            **0** |


## Record this milestone

| Metric                          |            Value |
| ------------------------------- | ---------------: |
| Input edges                     |      **111,090** |
| Input nodes                     |       **88,541** |
| Invalid/empty geometries        |            **0** |
| Self-loops removed              |           **24** |
| Short edges (≤ 0.5 m) removed   |        **1,027** |
| Invalid node references removed |            **0** |
| **Total edges removed**         |        **1,051** |
| **Clean edges**                 |      **110,039** |
| **Nodes retained**              |       **88,476** |
| Clean network length            |  **7,011.46 km** |
| Average edge length             |      **63.72 m** |
| Minimum edge length             |     **0.5002 m** |
| Maximum edge length             |   **2,694.95 m** |
| Connected components            |          **237** |
| Largest component               | **87,244 nodes** |
| Largest component coverage      |       **98.54%** |
| Components ≤10 nodes            |          **210** |
| Components removed              |            **0** |


## Milestone 2 — Intersection + Snapping Validation COMPLETE

| Metric                                     |             Value |
| ------------------------------------------ | ----------------: |
| Clean edges                                |       **110,039** |
| Clean nodes                                |        **88,476** |
| Unique intersections                       |        **67,420** |
| Intersections already represented by nodes | **67,420 (100%)** |
| Intersections requiring edge splitting     |            **13** |
| Strong snap candidates ≤0.5 m              |         **3,674** |
| Snap candidates 0.5–2 m                    |        **15,695** |
| Unique candidate connections               |        **15,357** |
| Connected components                       |           **172** |
| Largest component                          |  **87,244 nodes** |
| Largest component coverage                 |        **98.61%** |


## The endpoint mapping is now working. But I would not call the graph final yet

| Metric                     |          Result |
| -------------------------- | --------------: |
| Clean input edges          |     **110,039** |
| Clean nodes                |      **88,476** |
| Raw snap candidates        |      **15,357** |
| Strong candidates accepted |       **2,184** |
| Rejected/review            |      **12,878** |
| Invalid endpoint refs      |         **295** |
| Already connected          |         **645** |
| Snap connectors added      |       **2,184** |
| Final edges                |     **112,223** |
| Final nodes                |      **88,476** |
| Network length             | **7,012.09 km** |
| Connected components       |         **144** |
| Largest component          |      **87,808** |
| Coverage                   |      **99.24%** |
| Self-loops                 |           **0** |
| Invalid geometries         |           **0** |


## Milestone: Final Road Graph Construction

| Item                       |                             Value |
| -------------------------- | --------------------------------: |
| Input clean edges          |                       **110,039** |
| Input clean nodes          |                        **88,476** |
| Snap candidates            |                        **15,357** |
| Accepted snap connections  |                            **33** |
| Final edges                |                       **110,072** |
| Final nodes                |                        **88,476** |
| Self-loops                 |                             **0** |
| Zero-length edges          |                             **0** |
| Largest component          |                  **87,294 nodes** |
| Largest component coverage |                        **98.66%** |
| Connected components       |                           **165** |
| Maximum node degree        |                             **5** |
| Network length             |                   **7,011.47 km** |
| Status                     | **Pending final snap validation** |


## ResQTwin — Geospatial Milestone Record

| Item                       |           Result |
| -------------------------- | ---------------: |
| Final nodes                |       **88,476** |
| Final edges                |      **110,072** |
| Accepted snap connections  |           **33** |
| Rejected/review candidates |       **15,324** |
| Connected components       |          **165** |
| Largest component          | **87,294 nodes** |
| Largest component coverage |       **98.66%** |
| Components ≤10 nodes       |          **140** |
| Total network length       |  **7,011.47 km** |
| Median edge length         |       **44.9 m** |
| Maximum edge length        |   **2,694.95 m** |
| Self-loops                 |            **0** |
| Invalid geometries         |            **0** |
| Invalid node references    |            **0** |
| Duplicate edge IDs         |            **0** |
| Suspicious accepted snaps  |            **0** |
| Shortest-path test         |         **PASS** |
| Routing readiness          |          **YES** |


## Record this milestone

| Item                 |                        Value |
| -------------------- | ---------------------------: |
| Final routing edges  |                  **110,072** |
| Final routing nodes  |                   **88,476** |
| Baseline speed       |                  **30 km/h** |
| Network length       |              **7,011.47 km** |
| Largest component    |             **87,294 nodes** |
| Coverage             |                   **98.66%** |
| Test route           |                 **42.79 km** |
| Baseline travel time |                **85.58 min** |
| Routing test         |                     **PASS** |
| Enriched graph       | `road_edges_routing.geojson` |
| Test route           |    `test_time_route.geojson` |
