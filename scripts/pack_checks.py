"""Synthetic multi-line shipment fixture shared by the local and Claude benchmark arms."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

from prompture.agents.tools_schema import ToolRegistry
from prompture.workflow import Graph, Handle, HandleType, Node

POLICY = """For every manifest ID return [decision, reason], using the first applicable rule:
cancelled -> hold/cancelled; missing scan or incomplete scan -> review/incomplete;
wrong carton -> hold/carton; wrong destination -> hold/destination; outdated label revision -> hold/label;
same scan ID with conflicting SKU or quantity -> review/conflict (ignore exact duplicate observations);
observed SKU quantities differ from this shipment's lines -> hold/contents; otherwise pass/match.
Sum quantities across separate scan IDs for the same SKU. Each shipment is independent, including split orders."""


def fixture(extra_lines: int = 0) -> tuple[dict, dict, dict]:
    if extra_lines < 0:
        raise ValueError("extra_lines must be nonnegative")
    manifests, scans = {}, {}
    # Independent expected results; do not generate the oracle with the reducer being evaluated.
    expected = {
        "S11": ["pass", "match"],
        "S28": ["hold", "contents"],
        "S32": ["hold", "carton"],
        "S47": ["hold", "label"],
        "S53": ["hold", "destination"],
        "S64": ["review", "incomplete"],
        "S71": ["pass", "match"],
        "S82": ["review", "conflict"],
        "S95": ["hold", "cancelled"],
        "S09": ["hold", "contents"],
        "S17": ["review", "incomplete"],
        "S23": ["pass", "match"],
    }
    for sid in expected:
        manifests[sid] = {
            "order": f"O{sid}",
            "carton": f"C{sid}",
            "destination": f"D{sid}",
            "revision": 2,
            "cancelled": False,
            "lines": {"mug-blue": 2, "book": 1},
        }
        scans[sid] = {
            "carton": f"C{sid}",
            "destination": f"D{sid}",
            "revision": 2,
            "complete": True,
            "items": [["a", "mug-blue", 1], ["b", "mug-blue", 1], ["c", "book", 1]],
        }
    scans["S28"]["items"].pop()
    scans["S32"]["carton"] = "CS28"
    scans["S47"]["revision"] = 1
    scans["S53"]["destination"] = "DS11"
    del scans["S64"]
    scans["S71"]["items"].append(["b", "mug-blue", 1])
    scans["S82"]["items"].append(["a", "mug-red", 1])
    manifests["S95"]["cancelled"] = True
    scans["S09"]["items"].append(["d", "book", 1])
    scans["S17"]["complete"] = False
    manifests["S23"]["order"] = manifests["S11"]["order"]
    manifests["S23"]["lines"] = {"journal": 1}
    scans["S23"]["items"] = [["a", "journal", 1]]
    # Increase the intermediate data without changing any independently graded outcome.
    for sid, manifest in manifests.items():
        for i in range(extra_lines):
            sku = f"accessory-{i:04}"
            qty = 1 + i % 5
            manifest["lines"][sku] = qty
            if sid in scans:
                scans[sid]["items"].append([f"x{i}", sku, qty])
    return manifests, scans, expected


def pack_decision(manifest: dict, scan: dict | None) -> list[str]:
    if manifest["cancelled"]:
        return ["hold", "cancelled"]
    if scan is None or not scan["complete"]:
        return ["review", "incomplete"]
    for field, reason in (("carton", "carton"), ("destination", "destination"), ("revision", "label")):
        if manifest[field] != scan[field]:
            return ["hold", reason]
    seen = {}
    quantities: Counter[str] = Counter()
    for scan_id, sku, qty in scan["items"]:
        if scan_id in seen:
            if seen[scan_id] != (sku, qty):
                return ["review", "conflict"]
            continue
        seen[scan_id] = sku, qty
        quantities[sku] += qty
    return ["pass", "match"] if dict(quantities) == manifest["lines"] else ["hold", "contents"]


def audit(manifests: dict, scans: dict) -> dict:
    """Check each carton against its current shipment manifest."""
    return {sid: pack_decision(row, scans.get(sid)) for sid, row in manifests.items()}


def tools_for(manifests: dict, scans: dict, reads: Counter) -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool
    def read_manifests(batch: str) -> dict:
        """Read all expected shipment manifests for batch B1, keyed by shipment ID."""
        if batch != "B1":
            raise ValueError("Unknown batch")
        reads["manifests"] += 1
        return deepcopy(manifests)

    @registry.tool
    def read_scans(batch: str) -> dict:
        """Read physical packing observations for batch B1, keyed by shipment ID; items are [scan ID, SKU, quantity]."""
        if batch != "B1":
            raise ValueError("Unknown batch")
        reads["scans"] += 1
        return deepcopy(scans)

    registry.register(audit)
    return registry


def pack_graph() -> Graph:
    graph = Graph(
        id="audit_pack_batch",
        name="Read fresh manifests and scans and apply the pack policy to a batch.",
        inputs=[Handle("batch", HandleType.TEXT)],
        outputs={"decisions": "{{check.value}}"},
    )
    for node, tool in (("orders", "read_manifests"), ("scans", "read_scans")):
        graph.add_node(Node(node, "tool", {"tool": tool, "args": {"batch": "{{inputs.batch}}"}}))
    graph.add_node(
        Node("check", "tool", {"tool": "audit", "args": {"manifests": "{{orders.value}}", "scans": "{{scans.value}}"}})
    )
    return graph
