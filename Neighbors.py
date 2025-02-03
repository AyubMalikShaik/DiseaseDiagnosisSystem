import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv('./Dataset/mydataset.csv')

def buildGraph():
    G = nx.Graph()
    symptoms = df.columns[1:]

    for _, row in df.iterrows():
        for symptom in symptoms:
            if row[symptom] == 1:
                for other_symptom in symptoms:
                    if row[other_symptom] == 1 and symptom != other_symptom:
                        G.add_edge(symptom, other_symptom)
    return G

def buildGraphWithWeights():
    G = nx.Graph()
    symptoms = df.columns[1:]

    for _, row in df.iterrows():
        for symptom in symptoms:
            if row[symptom] == 1:
                for other_symptom in symptoms:
                    if row[other_symptom] == 1 and symptom != other_symptom:
                        if G.has_edge(symptom, other_symptom):
                            G[symptom][other_symptom]['weight'] += 1
                        else:
                            G.add_edge(symptom, other_symptom, weight=1)
    return G

def find_directly_related_symptoms(G, symptom):
    return list(G.neighbors(symptom))

def find_symptoms_within_hops(G, symptom, hops):
    return list(nx.single_source_shortest_path_length(G, symptom, cutoff=hops).keys())

def find_most_frequent_symptoms(G, symptom, top_n=15):
    neighbors = G.neighbors(symptom)
    co_occurrence_counts = {n: G.degree(n) for n in neighbors}
    sorted_counts = sorted(co_occurrence_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_counts[:top_n]

def find_strongest_connections(G, symptom, top_n=15):
    if symptom in G:
        weighted_edges = G[symptom]
        sorted_edges = sorted(weighted_edges.items(), key=lambda x: x[1]['weight'], reverse=True)
        return sorted_edges[:top_n]
    return []

def find_symptom_communities(G):
    import community as community_louvain

    partition = community_louvain.best_partition(G)
    communities = {}
    for symptom, community_id in partition.items():
        communities.setdefault(community_id, []).append(symptom)
    return communities

def visualize_symptom_graph(G, symptom):
    subgraph = G.subgraph([symptom] + list(G.neighbors(symptom)))
    pos = nx.spring_layout(subgraph)
    plt.figure(figsize=(8, 8))
    nx.draw(subgraph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=1500, font_size=12)
    plt.show()

# Build the graph
def sub_graph(graph,symptom):
    G_sub = graph.subgraph([symptom] + list(graph.neighbors(symptom)))
    return G_sub


def visualize_3d_symptom_graph(G, symptom):
    subgraph = G.subgraph([symptom] + list(G.neighbors(symptom)))
    
    # Create a 3D layout for nodes
    pos = nx.spring_layout(subgraph, dim=3)  # 3D layout

    # Get x, y, z coordinates for each node
    x_nodes = [pos[node][0] for node in subgraph.nodes()]
    y_nodes = [pos[node][1] for node in subgraph.nodes()]
    z_nodes = [pos[node][2] for node in subgraph.nodes()]

    # Create the edges
    x_edges = []
    y_edges = []
    z_edges = []

    for edge in subgraph.edges():
        x_edges.append(pos[edge[0]][0])
        x_edges.append(pos[edge[1]][0])
        y_edges.append(pos[edge[0]][1])
        y_edges.append(pos[edge[1]][1])
        z_edges.append(pos[edge[0]][2])
        z_edges.append(pos[edge[1]][2])

    # Plot nodes and edges
    edge_trace = go.Scatter3d(x=x_edges, y=y_edges, z=z_edges,
                              mode='lines',
                              line=dict(width=1, color='#2973B2'),
                              opacity=0.5)

    # Add node trace with text labels
    node_trace = go.Scatter3d(x=x_nodes, y=y_nodes, z=z_nodes,
                              mode='markers+text',  # Added +text for labels
                              marker=dict(symbol='circle', size=12, color='#48A6A7',opacity=0.7),
                              text=[node for node in subgraph.nodes()],  # Display node names as text
                              textposition='top center',
                              textfont=dict(
                                  color='#384B70',  # Change this to any color you like (e.g., 'blue', 'green', '#FF5733')
                                  size=12,  # Adjust text size
                                  family="Arial"  # Set a font style
                              ),
                              # Position the text above the nodes
                              showlegend=False)

    fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title=f"3D Visualization of Symptoms related to {symptom}",
        showlegend=False,
        scene=dict(
            xaxis=dict(
                title="X-Axis",
                gridcolor="#2973B2",  # Change X-axis grid color
                showgrid=True       # Ensure the grid is visible
            ),
            yaxis=dict(
                title="Y-Axis",
                gridcolor="#2973B2",  
                showgrid=True
            ),
            zaxis=dict(
                title="Z-Axis",
                gridcolor="#2973B2",  
                showgrid=True
            ),
            bgcolor="#D4F6FF"  # Background color for the entire 3D scene
        ),
        margin=dict(t=0, b=0, l=0, r=0)
        )
    )
    return fig

def visualize_communities(G, subgraph):
    
    # Create a 3D layout for nodes
    pos = nx.spring_layout(subgraph, dim=3)  # 3D layout

    # Get x, y, z coordinates for each node
    x_nodes = [pos[node][0] for node in subgraph.nodes()]
    y_nodes = [pos[node][1] for node in subgraph.nodes()]
    z_nodes = [pos[node][2] for node in subgraph.nodes()]

    # Create the edges
    x_edges = []
    y_edges = []
    z_edges = []

    for edge in subgraph.edges():
        x_edges.append(pos[edge[0]][0])
        x_edges.append(pos[edge[1]][0])
        y_edges.append(pos[edge[0]][1])
        y_edges.append(pos[edge[1]][1])
        z_edges.append(pos[edge[0]][2])
        z_edges.append(pos[edge[1]][2])

    # Plot nodes and edges
    edge_trace = go.Scatter3d(x=x_edges, y=y_edges, z=z_edges,
                              mode='lines',
                              line=dict(width=1, color='#2973B2'),
                              opacity=0.5)

    # Add node trace with text labels
    node_trace = go.Scatter3d(x=x_nodes, y=y_nodes, z=z_nodes,
                              mode='markers+text',  # Added +text for labels
                              marker=dict(symbol='circle', size=12, color='#48A6A7',opacity=0.7),
                              text=[node for node in subgraph.nodes()],  # Display node names as text
                              textposition='top center',
                              textfont=dict(
                                  color='#384B70',  # Change this to any color you like (e.g., 'blue', 'green', '#FF5733')
                                  size=12,  # Adjust text size
                                  family="Arial"  # Set a font style
                              ),
                              # Position the text above the nodes
                              showlegend=False)

    fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title=f"3D Visualization of Symptoms related to symptom",
        showlegend=False,
        scene=dict(
            xaxis=dict(
                title="X-Axis",
                gridcolor="#2973B2",  # Change X-axis grid color
                showgrid=True       # Ensure the grid is visible
            ),
            yaxis=dict(
                title="Y-Axis",
                gridcolor="#2973B2",  
                showgrid=True
            ),
            zaxis=dict(
                title="Z-Axis",
                gridcolor="#2973B2",  
                showgrid=True
            ),
            bgcolor="#D4F6FF"  # Background color for the entire 3D scene
        ),
        margin=dict(t=0, b=0, l=0, r=0)
        )
    )
    return fig
def display_most_freq_symptoms(G,symptom,count):
    mostfreq=find_most_frequent_symptoms(G, symptom,count)
    symptom_names = [symptom for symptom, _ in mostfreq]
    sub= G.subgraph([symptom] + symptom_names)
    fig=visualize_3d_symptom_graph(sub, symptom)
    return fig
def display_strongly_connected_symptoms(G,symptom,count):
    mostfreq=find_strongest_connections(G, symptom,count)
    symptom_names = [symptom for symptom, _ in mostfreq]
    sub= G.subgraph([symptom] + symptom_names)
    fig=visualize_3d_symptom_graph(sub, symptom)
    return fig
# Example usage
# symptom = 'cough'
G = buildGraph()
G = buildGraphWithWeights()
# fig=display_strongly_coumminicated_symptoms(G,symptom,10)
# fig.show()
#     # Make sure the graph is built with weights if needed
# sub = sub_graph(G,symptom)


# Example usage
# print("Directly related symptoms:", find_directly_related_symptoms(G, symptom))
# print("Symptoms within 2 hops:", find_symptoms_within_hops(G, symptom, 2))
# print("Most frequent symptoms:", find_most_frequent_symptoms(G, symptom,10))
# print("Strongest connections:", find_strongest_connections(G, symptom,10))
# visualize_symptom_graph(G, symptom)

# Find communities
communities = find_symptom_communities(G)
print(communities)
# for key,value in communities.items():
#     print(f"{value}")

