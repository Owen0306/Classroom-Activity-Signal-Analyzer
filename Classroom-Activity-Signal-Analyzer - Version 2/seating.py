# Optional seat regions for fixed-camera installations of any size or layout.

from collections import defaultdict
import json
import math
from pathlib import Path


class SeatMap:
    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("Seat map must be a JSON object")
        self.seats = data.get("seats", [])
        if not isinstance(self.seats, list) or not self.seats:
            raise ValueError("Seat map must contain a nonempty seats list")
        ids = set()
        for seat in self.seats:
            if not isinstance(seat, dict):
                raise ValueError("Each seat must be a JSON object")
            seat_id = str(seat.get("id", "")).strip()
            region = seat.get("region", [])
            if not seat_id or seat_id in ids:
                raise ValueError("Seat IDs must be nonempty and unique")
            if (not isinstance(region, list) or len(region) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                           and math.isfinite(v) and 0 <= v <= 1 for v in region)):
                raise ValueError("Seat regions must be [x1, y1, x2, y2] in normalized coordinates")
            if region[0] >= region[2] or region[1] >= region[3]:
                raise ValueError("Seat regions must have positive area")
            seat["id"] = seat_id
            ids.add(seat_id)

    def assign(self, observations, image_size):
        width, height = image_size
        claims = defaultdict(list)
        for observation in observations:
            observation.seat_id = None
            observation.seat_assignment = "outside_regions"
            x, y = observation.anchor
            x, y = x / width, y / height
            candidates = [s["id"] for s in self.seats
                          if s["region"][0] <= x < s["region"][2]
                          and s["region"][1] <= y < s["region"][3]]
            if len(candidates) == 1:
                claims[candidates[0]].append(observation)
            elif len(candidates) > 1:
                observation.seat_assignment = "ambiguous_regions"
        for seat_id, claimants in claims.items():
            if len(claimants) == 1:
                claimants[0].seat_id = seat_id
                claimants[0].seat_assignment = "assigned"
            else:
                for observation in claimants:
                    observation.seat_assignment = "multiple_detections_for_seat"
        assigned = {o.seat_id for o in observations if o.seat_id is not None}
        return [s["id"] for s in self.seats if s["id"] not in assigned]
