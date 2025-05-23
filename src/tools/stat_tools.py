from collections import Counter
from typing import List, Dict, Any


def compute_statistics(elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a list of enriched element dicts (each having element["analysis"]),
    compute summary stats: total, sensitive/public/private counts,
    counts per zone, camera_type, and top operators.
    :param elements: The element dict
    :return: Summary statistics
    """
    total = len(elements)
    analysis = [el["analysis"] for el in elements]

    sensitive_count = sum(1 for a in analysis if a.get("sensitive"))
    public_count = sum(1 for a in analysis if a.get("public") is True)
    private_count = sum(1 for a in analysis if a.get("public") is False)

    zone_counts = Counter(a.get("zone") for a in analysis if a.get("zone"))
    camera_type_counts = Counter(
        a.get("camera_type") for a in analysis if a.get("camera_type")
    )
    operator_counts = Counter(a.get("operator") for a in analysis if a.get("operator"))

    return {
        "total": total,
        "sensitive_count": sensitive_count,
        "public_count": public_count,
        "private_count": private_count,
        "zone_counts": zone_counts,
        "camera_type_counts": camera_type_counts,
        "operator_counts": operator_counts,
    }
