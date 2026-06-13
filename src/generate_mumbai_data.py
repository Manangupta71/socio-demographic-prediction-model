import os
import numpy as np
import pandas as pd

RNG_SEED = 42

# ---------------------------------------------------------------------------
# 1. Real-ish Mumbai ward / suburb reference geography
#    (approximate centroids, good enough for synthetic spatial structure)
# ---------------------------------------------------------------------------
MUMBAI_WARDS = {
    "Colaba":        (18.9067, 72.8147),
    "Fort":          (18.9345, 72.8359),
    "Dadar":         (19.0178, 72.8478),
    "Worli":         (19.0176, 72.8166),
    "Bandra":        (19.0596, 72.8295),
    "Khar":          (19.0696, 72.8266),
    "Santacruz":     (19.0840, 72.8367),
    "Andheri East":  (19.1136, 72.8697),
    "Andheri West":  (19.1197, 72.8468),
    "Jogeshwari":    (19.1364, 72.8478),
    "Goregaon":      (19.1646, 72.8493),
    "Malad":         (19.1864, 72.8484),
    "Borivali":      (19.2307, 72.8567),
    "Dahisar":       (19.2540, 72.8590),
    "Kurla":         (19.0726, 72.8826),
    "Ghatkopar":     (19.0857, 72.9081),
    "Vikhroli":      (19.1079, 72.9272),
    "Powai":         (19.1197, 72.9050),
    "Mulund":        (19.1726, 72.9570),
    "Bhandup":       (19.1485, 72.9357),
    "Chembur":       (19.0522, 72.9005),
    "Govandi":       (19.0550, 72.9270),
    "Mankhurd":      (19.0490, 72.9330),
    "Sion":          (19.0430, 72.8620),
}
WARD_NAMES = list(MUMBAI_WARDS.keys())

# Designate a subset of wards as "high commercial density" (CBD-like) so that
# work/anchor locations cluster realistically (Fort/BKC/Andheri-E etc.)
COMMERCIAL_WARDS = ["Fort", "Worli", "Bandra", "Andheri East", "Powai", "Chembur"]

# Wards near suburban rail / metro corridor -> baseline higher transit proximity
RAIL_PROXIMATE_WARDS = {
    "Dadar": 0.9, "Kurla": 0.85, "Ghatkopar": 0.8, "Andheri East": 0.85,
    "Andheri West": 0.6, "Bandra": 0.75, "Khar": 0.6, "Santacruz": 0.65,
    "Borivali": 0.7, "Malad": 0.65, "Goregaon": 0.6, "Mulund": 0.6,
    "Bhandup": 0.6, "Vikhroli": 0.55, "Chembur": 0.55, "Sion": 0.7,
    "Fort": 0.8, "Colaba": 0.4, "Worli": 0.35, "Jogeshwari": 0.55,
    "Dahisar": 0.5, "Mankhurd": 0.5, "Govandi": 0.5, "Powai": 0.3,
}


def sample_demographics(rng, n):
    """
    Sample joint demographic attributes from realistic marginal distributions
    with mild, plausible inter-attribute dependence (age <-> work status <->
    income <-> car ownership), but NOT a single deterministic archetype.
    """
    age_group = rng.choice(
        ["0-22", "22-60", "60_and_above"], size=n, p=[0.28, 0.62, 0.10]
    )
    gender = rng.choice(["Male", "Female"], size=n, p=[0.52, 0.48])

    work_status = np.empty(n, dtype=object)
    income_bracket = np.empty(n, dtype=object)
    household_auto = np.empty(n, dtype=object)
    driving_license = np.empty(n, dtype=object)

    for i in range(n):
        a = age_group[i]
        if a == "0-22":
            work_status[i] = rng.choice(
                ["Steady_Jobs_or_School", "Unemployed_or_Retired"], p=[0.75, 0.25]
            )
            income_bracket[i] = rng.choice(
                ["Low_Income", "Middle_High_Income"], p=[0.70, 0.30]
            )
            driving_license[i] = rng.choice(["Yes", "No"], p=[0.15, 0.85])
        elif a == "22-60":
            work_status[i] = rng.choice(
                ["Steady_Jobs_or_School", "Unemployed_or_Retired"], p=[0.78, 0.22]
            )
            if work_status[i] == "Steady_Jobs_or_School":
                income_bracket[i] = rng.choice(
                    ["Low_Income", "Middle_High_Income"], p=[0.45, 0.55]
                )
            else:
                income_bracket[i] = rng.choice(
                    ["Low_Income", "Middle_High_Income"], p=[0.80, 0.20]
                )
            driving_license[i] = rng.choice(["Yes", "No"], p=[0.55, 0.45])
        else:  # 60_and_above
            work_status[i] = rng.choice(
                ["Steady_Jobs_or_School", "Unemployed_or_Retired"], p=[0.15, 0.85]
            )
            income_bracket[i] = rng.choice(
                ["Low_Income", "Middle_High_Income"], p=[0.55, 0.45]
            )
            driving_license[i] = rng.choice(["Yes", "No"], p=[0.45, 0.55])

        # Car ownership correlates with income, weakly with driving license
        if income_bracket[i] == "Middle_High_Income":
            household_auto[i] = rng.choice(["0 cars", "1_or_more"], p=[0.35, 0.65])
        else:
            household_auto[i] = rng.choice(["0 cars", "1_or_more"], p=[0.80, 0.20])

    # Socio-demographic class: a coarse composite label derived probabilistically
    # from age/work/income (mimics a survey-coded segmentation variable),
    # with injected label noise so it's not a deterministic function.
    socio_class = np.empty(n, dtype=object)

    for i in range(n):

        if age_group[i] == "0-22":
            socio_class[i] = rng.choice(
            ["Type_a", "Type_c"],
            p=[0.9, 0.1]
        )

        elif age_group[i] == "60_and_above":
            socio_class[i] = rng.choice(
            ["Type_e", "Type_d"],
            p=[0.9, 0.1]
        )

        elif (
            income_bracket[i] == "Middle_High_Income"
            and household_auto[i] == "1_or_more"
    ):
            socio_class[i] = rng.choice(
            ["Type_b", "Type_c"],
            p=[0.9, 0.1]
        )

        elif work_status[i] == "Unemployed_or_Retired":
            socio_class[i] = rng.choice(
            ["Type_d", "Type_e"],
            p=[0.9, 0.1]
        )

        else:
            socio_class[i] = rng.choice(
            ["Type_c", "Type_b"],
            p=[0.9, 0.1]
        )

    return pd.DataFrame({
        "socio_demographic_class": socio_class,
        "age_group": age_group,
        "gender": gender,
        "work_status": work_status,
        "household_auto": household_auto,
        "driving_license": driving_license,
        "income_bracket": income_bracket,
    })


def assign_residential_ward(rng, demo_df):
    """
    Assign a home ward per individual. Income/car-ownership mildly bias the
    ward distribution (e.g. higher income skews towards South/Western suburbs
    like Bandra/Worli/Powai; lower income skews towards Eastern suburbs like
    Kurla/Govandi/Mankhurd/Bhandup) -- again probabilistic, not deterministic.
    """
    n = len(demo_df)
    high_income_skew_wards = ["Bandra", "Khar", "Santacruz", "Worli", "Powai",
                               "Andheri West", "Colaba", "Fort"]
    low_income_skew_wards = ["Kurla", "Govandi", "Mankhurd", "Bhandup",
                              "Vikhroli", "Jogeshwari", "Dahisar", "Malad"]

    base_p = np.ones(len(WARD_NAMES))
    wards = np.empty(n, dtype=object)
    for i in range(n):
        p = base_p.copy()
        if demo_df.loc[i, "income_bracket"] == "Middle_High_Income":
            for w in high_income_skew_wards:
                p[WARD_NAMES.index(w)] *= 3.0
        else:
            for w in low_income_skew_wards:
                p[WARD_NAMES.index(w)] *= 2.5
        p = p / p.sum()
        wards[i] = rng.choice(WARD_NAMES, p=p)
    return wards


def generate_cdr_logs(rng, phone_number, home_ward, demo_row, n_days=14, contact_pool=None):
    """
    Generate a per-subscriber raw CDR-style log combining:
      (a) spatial ping records (location, tower, near_railway_node), as before
      (b) communication events (calls/texts, direction, duration, contact id,
          hour, weekday) -- this second stream is what enables the literature's
          top-performing features: interevent time, nocturnal %, response
          delay/rate proxies, contact entropy/balance, percent initiated.

    Demographic -> behavior mappings (continuous propensities + noise, so no
    deterministic leakage):
      - nocturnal_propensity:  fraction of communication after 22:00 or before
        6:00. Higher for younger / student population (Type 'a'-like ages),
        lower for older/working professionals with early schedules.
      - call_volume: overall daily interaction count, driven by age + work
        status + income (richer / working-age people communicate more).
      - initiation_bias: probability the user initiates vs receives -- driven
        by income (higher income -> initiates more, less cost-sensitive) and
        gender (small effect, mirroring literature finding that this is
        culturally/economically driven rather than a strong direct gender
        signal).
      - response_delay_scale: how quickly the user responds to texts -- driven
        by work status (employed people respond faster during day) and age.
      - contact_diversity: number of distinct contacts -- driven by work
        status and age (socially active working-age adults have more contacts).

    Returns a dict with two DataFrames: {'pings': ..., 'comm': ...}
    """
    home_lat, home_lon = MUMBAI_WARDS[home_ward]

    age, work, income, auto, gender = (demo_row["age_group"], demo_row["work_status"],
                                        demo_row["income_bracket"], demo_row["household_auto"],
                                        demo_row["gender"])

    # --- continuous "propensity" scores from demographics, each in [0,1] ---
    mobility = 0.3
    if work == "Steady_Jobs_or_School":
        mobility += 0.35
    if age == "22-60":
        mobility += 0.15
    elif age == "60_and_above":
        mobility -= 0.15
    if income == "Middle_High_Income":
        mobility += 0.1
    mobility = np.clip(mobility + rng.normal(0, 0.15), 0.05, 1.0)

    transit_affinity = RAIL_PROXIMATE_WARDS.get(home_ward, 0.5)
    if auto == "1_or_more":
        transit_affinity -= 0.15
    if income == "Middle_High_Income" and auto == "1_or_more":
        transit_affinity -= 0.10
    transit_affinity = np.clip(transit_affinity + rng.normal(0, 0.15), 0.05, 0.95)

    if work == "Unemployed_or_Retired":
        n_anchors = rng.choice([0, 1], p=[0.55, 0.45])
    else:
        n_anchors = rng.choice([1, 2, 3], p=[0.45, 0.40, 0.15])

    anchors = []
    for _ in range(n_anchors):
        if rng.random() < transit_affinity:
            anchor_ward = rng.choice(COMMERCIAL_WARDS)
        else:
            anchor_ward = rng.choice(WARD_NAMES)
        a_lat, a_lon = MUMBAI_WARDS[anchor_ward]
        spread = 0.01 + 0.04 * mobility
        anchors.append((a_lat + rng.normal(0, spread), a_lon + rng.normal(0, spread), anchor_ward))

    # --- communication propensities ---
    nocturnal_propensity = 0.15
    if age == "0-22":
        nocturnal_propensity += 0.20
    elif age == "60_and_above":
        nocturnal_propensity -= 0.08
    if work == "Unemployed_or_Retired":
        nocturnal_propensity += 0.05
    nocturnal_propensity = np.clip(nocturnal_propensity + rng.normal(0, 0.08), 0.02, 0.6)

    call_volume_base = 8
    if work == "Steady_Jobs_or_School":
        call_volume_base += 6
    if age == "22-60":
        call_volume_base += 3
    if income == "Middle_High_Income":
        call_volume_base += 2
    call_volume_base = max(2, call_volume_base + rng.normal(0, 3))

    initiation_bias = 0.5
    if income == "Middle_High_Income":
        initiation_bias += 0.07
    if gender == "Male":
        initiation_bias += 0.08
    initiation_bias = np.clip(initiation_bias + rng.normal(0, 0.08), 0.15, 0.85)

    response_delay_scale = 25.0  # mean minutes
    if work == "Steady_Jobs_or_School":
        response_delay_scale -= 6
    if age == "60_and_above":
        response_delay_scale += 8
    response_delay_scale = max(3.0, response_delay_scale + rng.normal(0, 5))

    n_contacts_base = 8
    if work == "Steady_Jobs_or_School":
        n_contacts_base += 5
    if age == "22-60":
        n_contacts_base += 3
    if gender == "Female":
        n_contacts_base += 2
    n_contacts = max(2, int(n_contacts_base + rng.normal(0, 3)))

    # Sample this user's social contacts from a shared pool (so co-contact
    # structure exists), falling back to synthetic IDs if no pool given.
    if contact_pool is not None and len(contact_pool) > n_contacts:
        contacts = list(rng.choice(contact_pool, size=n_contacts, replace=False))
    else:
        contacts = [f"CONTACT_{phone_number[-4:]}_{i}" for i in range(n_contacts)]

    # Give each contact a relative "closeness" weight (Zipf-like), so a few
    # contacts dominate interaction share -> drives entropy/balance features.
    weights = np.array([1.0 / (i + 1) for i in range(n_contacts)])
    weights = weights * rng.uniform(0.7, 1.3, size=n_contacts)
    weights = weights / weights.sum()

    records = []  # spatial pings
    comm_records = []  # communication events
    tower_id_cache = {}

    def tower_for(lat, lon):
        key = (round(lat, 3), round(lon, 3))
        if key not in tower_id_cache:
            tower_id_cache[key] = f"TWR_{len(tower_id_cache) + len(MUMBAI_WARDS)}_{phone_number[-4:]}"
        return tower_id_cache[key]

    home_tower = f"TWR_HOME_{home_ward.replace(' ', '')}"

    for day in range(n_days):
        weekday = day % 7 < 5  # Mon-Fri

        # --- spatial pings (unchanged logic) ---
        n_pings = rng.integers(8, 20)
        for _ in range(n_pings):
            hour = rng.integers(0, 24)
            is_peak = (8 <= hour <= 10) or (17 <= hour <= 19)

            if anchors and weekday and (is_peak or (8 <= hour <= 18)):
                p_anchor = 0.55 if is_peak else 0.40
            elif anchors and not weekday:
                p_anchor = 0.15
            else:
                p_anchor = 0.05

            if anchors and rng.random() < p_anchor:
                a_lat, a_lon, _ = anchors[rng.integers(0, len(anchors))]
                lat = a_lat + rng.normal(0, 0.005)
                lon = a_lon + rng.normal(0, 0.005)
                tower_id = tower_for(lat, lon)
                near_rail = 1 if rng.random() < transit_affinity else 0
            else:
                spread = 0.005 + 0.02 * mobility
                lat = home_lat + rng.normal(0, spread)
                lon = home_lon + rng.normal(0, spread)
                tower_id = home_tower if rng.random() < 0.6 else tower_for(lat, lon)
                near_rail = 1 if rng.random() < transit_affinity * 0.8 else 0

            timestamp = pd.Timestamp("2024-01-01")+ pd.Timedelta(days=day, hours=int(hour),
                                                                     minutes=int(rng.integers(0, 60)))
            records.append([phone_number, timestamp, lat, lon, tower_id, near_rail])

        # --- communication events ---
        n_events = max(0, int(rng.poisson(call_volume_base)))
        for _ in range(n_events):
            # Hour sampled from a mixture: daytime base + nocturnal_propensity
            if rng.random() < nocturnal_propensity:
                hour = int(rng.choice([22, 23, 0, 1, 2, 3, 4, 5]))
            else:
                # daytime activity, peaking late morning / evening
                hour = int(np.clip(rng.normal(14, 4), 6, 21))

            minute = int(rng.integers(0, 60))
            timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day, hours=hour, minutes=minute)

            event_type = "text" if rng.random() < 0.6 else "call"
            direction = "out" if rng.random() < initiation_bias else "in"

            contact_id = rng.choice(contacts, p=weights)

            if event_type == "call":
                # call duration in seconds; employed users tend to have
                # shorter, more purposeful calls during the day
                base_dur = 180
                if gender == "Female":
                    base_dur *= 1.2
                if 9 <= hour <= 18 and work == "Steady_Jobs_or_School":
                    base_dur = 90
                duration = max(5, int(rng.exponential(base_dur)))
            else:
                duration = 0

            # Response delay only meaningful for incoming texts that get a
            # reply -- we simulate a delay value attached to the event itself
            # representing "time to respond to previous incoming message"
            response_delay_min = max(0.5, rng.exponential(response_delay_scale))

            comm_records.append([
                phone_number, timestamp, event_type, direction, contact_id,
                duration, response_delay_min, day, weekday
            ])

    pings_df = pd.DataFrame(records, columns=[
        "phone_number", "timestamp", "latitude", "longitude", "tower_id", "near_railway_node"
    ])
    comm_df = pd.DataFrame(comm_records, columns=[
        "phone_number", "timestamp", "event_type", "direction", "contact_id",
        "duration_sec", "response_delay_min", "day", "weekday"
    ])
    return {"pings": pings_df, "comm": comm_df}


def generate_full_schema_mumbai_dataset(n_samples=1000, write_raw_logs=True, n_days=14):
    print("\n" + "=" * 70)
    print("   GENERATING MUMBAI SYNTHETIC POPULATION (SURVEY + CDR LOGS)")
    print("=" * 70)

    rng = np.random.default_rng(RNG_SEED)
    phone_numbers = [f"+919820{i:06d}" for i in range(n_samples)]

    # 1. Demographics (independent of mobility features at generation time)
    demo_df = sample_demographics(rng, n_samples)

    # 2. Residential ward, mildly correlated with income/auto ownership
    wards = assign_residential_ward(rng, demo_df)
    demo_df.insert(0, "residential_ward", wards)
    demo_df.insert(0, "phone_number", phone_numbers)

    os.makedirs("raw_data", exist_ok=True)
    if write_raw_logs:
        os.makedirs("raw_data/subscriber_logs", exist_ok=True)

    # 3. Generate per-subscriber CDR logs (and also a pre-aggregated fallback
    #    feature table using the same extraction logic as pipeline.py would,
    #    so the pipeline can run even if logs are skipped).
    from pipeline import extract_bandicoot_spatial_features  # local import to avoid cycle at module load

    feature_rows = []
    for i, phone in enumerate(phone_numbers):
        demo_row = demo_df.iloc[i]
        cdr_logs = generate_cdr_logs(
        rng,
        phone,
        demo_row["residential_ward"],
        demo_row,
        n_days=n_days
)

        pings_df = cdr_logs["pings"]
        comm_df = cdr_logs["comm"]

        if write_raw_logs:

            os.makedirs("raw_data/subscriber_logs", exist_ok=True)
            os.makedirs("raw_data/subscriber_comm", exist_ok=True)

            pings_df.to_csv(
                f"raw_data/subscriber_logs/{phone}.csv",
                index=False
           )

            comm_df.to_csv(
                f"raw_data/subscriber_comm/{phone}.csv",
                index=False
            )

    feats = extract_bandicoot_spatial_features(pings_df)
        
    feature_rows.append([phone] + feats)

    if (i + 1) % 200 == 0:
            print(f"  ...generated CDR logs for {i + 1}/{n_samples} subscribers")

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
    "hour_entropy",
    "unique_contacts",
    "contact_entropy",
]
    df_features = pd.DataFrame(feature_rows, columns=feature_cols)

    df_features.to_csv("raw_data/mumbai_phone_features.csv", index=False)
    demo_df.to_csv("raw_data/mumbai_survey.csv", index=False)

    # 4. Write ward reference table (for spatial joins / mapping later)
    ward_df = pd.DataFrame([
        {"ward": w, "latitude": lat, "longitude": lon}
        for w, (lat, lon) in MUMBAI_WARDS.items()
    ])
    ward_df.to_csv("raw_data/mumbai_wards.csv", index=False)

    print(f"\n[SUCCESS] Wrote {n_samples} records:")
    print("  raw_data/mumbai_survey.csv          (demographics + residential_ward)")
    print("  raw_data/mumbai_phone_features.csv  (pre-aggregated CDR features, fallback)")
    if write_raw_logs:
        print("  raw_data/subscriber_logs/<phone>.csv (raw per-subscriber CDR ping logs)")
    print("  raw_data/mumbai_wards.csv            (ward name -> lat/lon centroid reference)")


if __name__ == "__main__":
    # write_raw_logs=True writes ~1000 per-subscriber CSVs (14 days of pings each).
    # Set to False for a quick run that only produces the aggregated feature table.
    generate_full_schema_mumbai_dataset(n_samples=1000, write_raw_logs=True, n_days=14)