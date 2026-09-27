"""Observed display state; intentionally separate from model input defaults."""


def scoreboard_context(row: dict, stamp: str) -> dict:
    ball = str(row.get("possession") or "")
    # The yard-line abbreviation is NOT proof of possession.
    possession = ("home" if ball and ball == row["home_id"] else
                  "away" if ball and ball == row["away_id"] else None)
    down, distance = row.get("down"), row.get("distance")
    return {
        "updated_at": stamp, "possession": possession,
        "down": down if down is not None and 1 <= down <= 4 else None,
        "distance": distance if distance is not None and distance >= 0 else None,
        "down_distance": row.get("down_distance") or None,
        "field_position": row.get("possession_text") or None,
        "status_name": row.get("status_name") or None,
        "status_text": row.get("status_text") or None,
    }
