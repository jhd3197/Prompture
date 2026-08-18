"""Node-graph (DAG) workflow engine.

Build a typed graph of nodes (LLM, media generation, tool, transform), wire them
with explicit edges and/or late-bound ``{{node.outputs[0].value}}`` references,
and run it — full graph, a single node, or compiled to a callable.

Example::

    from prompture.workflow import Graph, Node, Handle, HandleType, compile_graph

    g = Graph(id="brief", inputs=[Handle("topic", HandleType.TEXT)])
    g.add_node(Node(id="topic", type="input"))
    g.add_node(Node(id="summary", type="llm", config={
        "model": "openai/gpt-4o-mini",
        "prompt": "One-line brief about {{topic.outputs.value}}",
    }))
    g.outputs = {"brief": "{{summary.outputs.text}}"}

    brief = compile_graph(g)
    out = brief(topic="an AI workflow engine")   # {"brief": "..."}
"""

from __future__ import annotations

from .architect import (
    ArchitectResult,
    EdgeSpec,
    NodeSpec,
    WorkflowGraphSpec,
    aarchitect,
    architect,
    node_type_catalog,
)
from .engine import (
    CompiledWorkflow,
    GraphRunner,
    compile_graph,
    run_graph,
    run_node,
)
from .errors import (
    CycleError,
    GraphValidationError,
    HandleTypeError,
    NodeExecutionError,
    ReferenceResolutionError,
    WorkflowError,
)
from .model import Edge, Graph, Handle, HandleType, Node, build_graph
from .references import (
    Reference,
    extract_dependencies,
    extract_refs,
    resolve_template,
)
from .registry import (
    NodeType,
    get_node_type,
    is_node_type_registered,
    list_node_types,
    register_builtin_node_types,
    register_node_type,
    unregister_node_type,
)
from .state import (
    NodeResult,
    NodeResultStore,
    NodeStatus,
    RunResult,
    RunStatus,
    WorkflowCallbacks,
)

__all__ = [
    "ArchitectResult",
    "CompiledWorkflow",
    "CycleError",
    "Edge",
    "EdgeSpec",
    # model
    "Graph",
    # engine
    "GraphRunner",
    "GraphValidationError",
    "Handle",
    "HandleType",
    "HandleTypeError",
    "Node",
    "NodeExecutionError",
    # state
    "NodeResult",
    "NodeResultStore",
    "NodeSpec",
    "NodeStatus",
    # registry
    "NodeType",
    # references
    "Reference",
    "ReferenceResolutionError",
    "RunResult",
    "RunStatus",
    "WorkflowCallbacks",
    # errors
    "WorkflowError",
    "WorkflowGraphSpec",
    "aarchitect",
    "architect",
    "build_graph",
    "compile_graph",
    "extract_dependencies",
    "extract_refs",
    "get_node_type",
    "is_node_type_registered",
    "list_node_types",
    "node_type_catalog",
    "register_builtin_node_types",
    "register_node_type",
    "resolve_template",
    "run_graph",
    "run_node",
    "unregister_node_type",
]

# Register built-in node types at import (mirrors register_all_builtin_drivers()).
register_builtin_node_types()
