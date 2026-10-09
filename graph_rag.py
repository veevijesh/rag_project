import networkx as nx

# 1. Initialize Knowledge Graph
G = nx.DiGraph()

# 2. Add Entities (Nodes)
G.add_node("Alice", type="Employee", role="Senior Developer")
G.add_node("Bob", type="Manager", role="Engineering Lead")
G.add_node("Remote Work Policy", type="Policy", details="2 days/week")
G.add_node("Internet Reimbursement", type="Expense", limit="$50/month")

# 3. Add Relationships (Edges)
G.add_edge("Alice", "Bob", relation="REPORTS_TO")
G.add_edge("Bob", "Remote Work Policy", relation="APPROVES_REQUESTS_FOR")
G.add_edge("Alice", "Internet Reimbursement", relation="SUBMITTED_CLAIM_FOR")


def graph_search(entity: str):
    """Explores multi-hop relationships starting from a given entity."""
    if entity not in G:
        print(f"Entity '{entity}' not found in Knowledge Graph.")
        return

    print(f"\n🔍 [Graph Traversal]: Exploring connections for '{entity}'...")
    
    # 1-Hop Connections
    for neighbor in G.neighbors(entity):
        relation = G[entity][neighbor]["relation"]
        print(f"  -> ({entity}) --[{relation}]--> ({neighbor})")
        
        # 2-Hop Connections (Multi-hop reasoning across nodes)
        for sub_neighbor in G.neighbors(neighbor):
            sub_relation = G[neighbor][sub_neighbor]["relation"]
            print(f"      ---> ({neighbor}) --[{sub_relation}]--> ({sub_neighbor})")


if __name__ == "__main__":
    graph_search("Alice")