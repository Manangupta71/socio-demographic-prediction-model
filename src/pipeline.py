"""
Feature extraction + graph construction pipeline.
"""

import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def extract_bandicoot_spatial_features(raw_cdr_df):
    if raw_cdr_df.empty:
        return [
        0.0, 0.0, 0.5, 0.4, 1,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0
    ]
    for col in ["data_volume_mb", "event_type", "duration_seconds"]:
        if col not in raw_cdr_df.columns:
            raw_cdr_df[col] = "" if col == "event_type" else 0

    lat_cm = raw_cdr_df["latitude"].mean()
    lon_cm = raw_cdr_df["longitude"].mean()
    distances_squared = (raw_cdr_df["latitude"] - lat_cm) ** 2 + (raw_cdr_df["longitude"] - lon_cm) ** 2
    radius_of_gyration = np.sqrt(distances_squared.mean())

    node_probabilities = raw_cdr_df["tower_id"].value_counts(normalize=True).values
    location_entropy = -np.sum(node_probabilities * np.log2(node_probabilities + 1e-9))
    
    raw_cdr_df = raw_cdr_df.copy()
    raw_cdr_df["timestamp"] = pd.to_datetime(raw_cdr_df["timestamp"])
    hours = raw_cdr_df["timestamp"].dt.hour

    peak_pings = ((hours >= 8) & (hours <= 10)) | ((hours >= 17) & (hours <= 19))
    off_peak_pings = (hours >= 12) & (hours <= 15)

    total_relevant = peak_pings.sum() + off_peak_pings.sum()
    peak_commute_ratio = peak_pings.sum() / total_relevant if total_relevant > 0 else 0.5

    transit_proximity = raw_cdr_df["near_railway_node"].mean() if "near_railway_node" in raw_cdr_df.columns else 0.4
    unique_anchors = raw_cdr_df["tower_id"].nunique()

    # Transactional features
    total_data_volume = raw_cdr_df["data_volume_mb"].sum()
    
    voice_df = raw_cdr_df[raw_cdr_df["event_type"].str.contains("voice", na=False)]
    total_voice_duration = voice_df["duration_seconds"].sum()
    
    sms_count = raw_cdr_df["event_type"].str.contains("sms", na=False).sum()
    voice_count = len(voice_df)
    sms_to_voice_ratio = sms_count / (voice_count + 1)
    
    # Social-network features
    if "opposite_party_number" in raw_cdr_df.columns:

        unique_contacts = (
        raw_cdr_df["opposite_party_number"]
        .astype(str)
        .nunique()
    )

        contact_probs = (
        raw_cdr_df["opposite_party_number"]
        .astype(str)
        .value_counts(normalize=True)
        .values
    )

        contact_entropy = -np.sum(
        contact_probs * np.log2(contact_probs + 1e-9)
    )

    else:
        unique_contacts = 0
        contact_entropy = 0
    
    active_events = raw_cdr_df["event_type"].isin(["voice_in", "voice_out", "sms_in", "sms_out", "data_session"]).sum()
    activity_ratio = active_events / len(raw_cdr_df) if len(raw_cdr_df) > 0 else 0.0
    
    # Temporal features
    night_events = ((hours >= 22) | (hours <= 5)).sum()
    night_activity_ratio = night_events / len(raw_cdr_df)

    weekend_events = raw_cdr_df["timestamp"].dt.dayofweek.isin([5, 6]).sum()
    weekend_ratio = weekend_events / len(raw_cdr_df)

# Communication features
    avg_voice_duration = (
      total_voice_duration / voice_count
      if voice_count > 0 else 0
    )

    voice_activity_density = voice_count / len(raw_cdr_df)
    sms_activity_density = sms_count / len(raw_cdr_df)

# Behavioral regularity
    active_days = raw_cdr_df["timestamp"].dt.date.nunique()

    hour_probs = hours.value_counts(normalize=True).values
    hour_entropy = -np.sum(hour_probs * np.log2(hour_probs + 1e-9))
    return [
    radius_of_gyration,
    location_entropy,
    peak_commute_ratio,
    transit_proximity,
    unique_anchors,

    total_data_volume,
    total_voice_duration,
    sms_to_voice_ratio,
    activity_ratio,

    night_activity_ratio,
    weekend_ratio,
    avg_voice_duration,
    voice_activity_density,
    sms_activity_density,
    active_days,
    hour_entropy,

    unique_contacts,
    contact_entropy
]

def build_colocation_graph(raw_log_files, max_files=None, top_k_edges_per_node=8):
    if max_files is not None:
        raw_log_files = raw_log_files[:max_files]

    phone_towers = {}
    for fp in raw_log_files:
        phone = os.path.basename(fp).replace(".csv", "")
        df = pd.read_csv(fp, usecols=["tower_id"])
        phone_towers[phone] = set(df["tower_id"].unique())

    tower_phones = {}
    for phone, towers in phone_towers.items():
        for t in towers:
            tower_phones.setdefault(t, []).append(phone)

    from collections import defaultdict
    pair_weight = defaultdict(int)
    for t, phones in tower_phones.items():
        if len(phones) < 2 or len(phones) > 50:
            continue
        for i in range(len(phones)):
            for j in range(i + 1, len(phones)):
                a, b = phones[i], phones[j]
                key = (a, b) if a < b else (b, a)
                pair_weight[key] += 1

    G = nx.Graph()
    for phone in phone_towers:
        G.add_node(phone)

    node_edges = defaultdict(list)
    for (a, b), w in pair_weight.items():
        node_edges[a].append((b, w))
        node_edges[b].append((a, w))

    seen = set()
    for node, edges in node_edges.items():
        edges.sort(key=lambda x: -x[1])
        for other, w in edges[:top_k_edges_per_node]:
            key = (node, other) if node < other else (other, node)
            if key in seen:
                continue
            seen.add(key)
            G.add_edge(node, other, weight=w)

    return G

def build_interaction_social_graph(raw_log_files):
    G = nx.Graph()

    for fp in raw_log_files:
        phone = os.path.basename(fp).replace(".csv", "")
        G.add_node(phone)

        try:
            df = pd.read_csv(fp)

            if "opposite_party_number" not in df.columns:
                continue

            interactions = (
                df["opposite_party_number"]
                .dropna()
                .astype(str)
            )

            for target in interactions:
                if target == phone:
                    continue

                if G.has_edge(phone, target):
                    G[phone][target]["weight"] += 1
                else:
                    G.add_edge(phone, target, weight=1)

        except Exception:
            continue

    return G

def build_mobility_similarity_graph(features_df, k=8):
    feature_cols = [
    "radius_of_gyration",
    "location_entropy",
    "peak_commute_ratio",
    "rail_transit_proximity",
    "unique_anchors",

    "total_data_volume_mb",
    "total_voice_duration_secs",
    "sms_to_voice_ratio",
    "activity_ratio",

    "night_activity_ratio",
    "weekend_ratio",
    "avg_voice_duration",
    "voice_activity_density",
    "sms_activity_density",
    "active_days",
    "hour_entropy"
] 
    X =StandardScaler().fit_transform(features_df[feature_cols].values)

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(features_df)))
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    G = nx.Graph()
    phones = features_df["phone_number"].values
    for p in phones:
        G.add_node(p)

    for i, phone in enumerate(phones):
        for j_idx in range(1, indices.shape[1]):
            j = indices[i, j_idx]
            other = phones[j]
            G.add_edge(phone, other, weight=1.0 / (1e-6 + distances[i, j_idx]))

    return G

def compute_graph_structural_features(G, phone_numbers):
    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    clustering = nx.clustering(G)

    try:
        communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    except Exception:
        communities = [set(G.nodes())]

    community_size = {}
    for comm in communities:
        for node in comm:
            community_size[node] = len(comm) / max(1, G.number_of_nodes())

    rows = []
    for phone in phone_numbers:
        rows.append({
            "phone_number": phone,
            "degree": degree.get(phone, 0),
            "weighted_degree": weighted_degree.get(phone, 0.0),
            "clustering_coef": clustering.get(phone, 0.0),
            "community_rel_size": community_size.get(phone, 1.0),
        })
    return pd.DataFrame(rows)

def run_feature_alignment_pipeline(use_graph_features=True, colocation_max_files=None):
    print("[PIPELINE] Initializing feature extraction runtime framework...")

    raw_logs_pattern = os.path.join("raw_data", "subscriber_logs", "*.csv")
    raw_log_files = sorted(glob.glob(raw_logs_pattern))

    if len(raw_log_files) > 0:
        print(f"[EXTRACT] Found {len(raw_log_files)} raw user logs. Computing metrics...")
        extracted_records = []
        for file_path in raw_log_files:
            phone_num = os.path.basename(file_path).replace(".csv", "")
            df_log = pd.read_csv(file_path)
            metrics = extract_bandicoot_spatial_features(df_log)
            extracted_records.append([phone_num] + metrics)

        feature_cols = [
    "phone_number",

    "radius_of_gyration",
    "location_entropy",
    "peak_commute_ratio",
    "rail_transit_proximity",
    "unique_anchors",

    "total_data_volume_mb",
    "total_voice_duration_secs",
    "sms_to_voice_ratio",
    "activity_ratio",

    "night_activity_ratio",
    "weekend_ratio",
    "avg_voice_duration",
    "voice_activity_density",
    "sms_activity_density",
    "active_days",
    "hour_entropy"
]
        features_df = pd.DataFrame(extracted_records, columns=feature_cols)
        features_df.to_csv("raw_data/mumbai_phone_features.csv", index=False)
    else:
        print("[PIPELINE] Raw subscriber logs missing. Importing fallback feature table.")
        master_feature_path = os.path.join("raw_data", "mumbai_phone_features.csv")
        if not os.path.exists(master_feature_path):
            raise FileNotFoundError(f"Missing feature matrix table asset at: {master_feature_path}")
        features_df = pd.read_csv(master_feature_path, dtype={"phone_number": str})
        features_df = features_df.fillna(features_df.median(numeric_only=True))

    if use_graph_features:
        phone_numbers = features_df["phone_number"].values

        print("[GRAPH] Building mobility-similarity graph (k-NN)...")
        sim_graph = build_mobility_similarity_graph(features_df, k=8)
        sim_feats = compute_graph_structural_features(sim_graph, phone_numbers)
        sim_feats = sim_feats.add_prefix("sim_").rename(columns={"sim_phone_number": "phone_number"})
        features_df = features_df.merge(sim_feats, on="phone_number", how="left")

        if len(raw_log_files) > 0:
            print("[GRAPH] Building CDR co-location graph...")
            colo_graph = build_colocation_graph(raw_log_files, max_files=colocation_max_files)
            colo_feats = compute_graph_structural_features(colo_graph, phone_numbers)
            colo_feats = colo_feats.add_prefix("colo_").rename(columns={"colo_phone_number": "phone_number"})
            features_df = features_df.merge(colo_feats, on="phone_number", how="left")

            print("[GRAPH] Building transactional social communication interaction graph...")
            social_graph = build_interaction_social_graph(raw_log_files)
            social_feats = compute_graph_structural_features(social_graph, phone_numbers)
            social_feats = social_feats.add_prefix("social_").rename(columns={"social_phone_number": "phone_number"})
            features_df = features_df.merge(social_feats, on="phone_number", how="left")

        features_df = features_df.fillna(0.0)

    phone_ids = features_df["phone_number"].values
    return phone_ids, features_df

if __name__ == "__main__":
    ids, df = run_feature_alignment_pipeline()
    print(f"\n[DONE] Feature matrix shape: {df.shape}")