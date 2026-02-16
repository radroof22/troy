"""Build execution graphs from agent traces."""

import networkx as nx

from troy.models import Trace


def build_execution_graph(trace: Trace) -> nx.DiGraph:
    """Build a directed graph from a trace's steps.

    Nodes are steps (keyed by step_id) with all step data as attributes.
    Edges represent sequential execution order, or parent-child relationships
    when parent_step_id is set.
    """
    graph = nx.DiGraph()

    for step in trace.steps:
        graph.add_node(
            step.step_id,
            type=step.type.value,
            description=step.description,
            input=step.input,
            output=step.output,
            metadata=step.metadata,
            timestamp=step.timestamp,
        )

    # Add edges: parent->child if parent_step_id set, otherwise sequential
    for i, step in enumerate(trace.steps):
        if step.parent_step_id and step.parent_step_id in graph:
            graph.add_edge(step.parent_step_id, step.step_id)
        elif i > 0:
            graph.add_edge(trace.steps[i - 1].step_id, step.step_id)

    return graph
