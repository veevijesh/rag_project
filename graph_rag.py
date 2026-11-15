"""A TOY knowledge graph. The nodes and edges are typed in by hand, not extracted from documents.

It shows why graphs help with multi-hop questions: the answer is a chain of facts
(Alice -> Bob -> Remote Work Policy) that never appears together in a single chunk of text.
"""
import networkx as nx

G = nx.DiGraph()

G.add_node("Alice", type="Employee", role="Senior Developer")
G.add_node("Bob", type="Manager", role="Engineering Lead")
G.add_node("Remote Work Policy", type="Policy", details="2 days/week")
G.add_node("Internet Reimbursement", type="Expense", limit="$50/month")

G.add_edge("Alice", "Bob", relation="REPORTS_TO")
G.add_edge("Bob", "Remote Work Policy", relation="APPROVES_REQUESTS_FOR")
G.add_edge("Alice", "Internet Reimbursement", relation="SUBMITTED_CLAIM_FOR")


def graph_search(entity, depth=2):
    """Print and return the connections within `depth` hops (breadth-first search).

    Outgoing edges are followed with BFS; incoming edges are listed separately,
    because a directed graph only follows edges in their own direction.
    """
    if entity not in G:
        print(f"Entity '{entity}' not found in the knowledge graph.")
        return []

    found = []
    print(f"\nOutgoing connections from '{entity}' (up to {depth} hops):")
    for source, target in nx.bfs_edges(G, entity, depth_limit=depth):
        relation = G[source][target]["relation"]
        found.append((source, relation, target))
        print(f"  ({source}) --[{relation}]--> ({target})")
    if not found:
        print("  (none)")

    print(f"Incoming connections to '{entity}':")
    incoming = list(G.predecessors(entity))
    for source in incoming:
        print(f"  ({source}) --[{G[source][entity]['relation']}]--> ({entity})")
    if not incoming:
        print("  (none)")
    return found


if __name__ == "__main__":
    graph_search("Alice")
    graph_search("Remote Work Policy")