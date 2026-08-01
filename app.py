import re
from copy import deepcopy

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Constants
# -----------------------------
POP_SIZE = 100
REP_FACTOR = 1800  # each icon ~1,800 real residents
SEED = 20260801

AGE_OPTIONS = ["0-17", "18-39", "40-64", "65+"]
LANG_OPTIONS = ["English", "Spanish", "Vietnamese", "Tagalog", "Other"]
ZIP_OPTIONS = ["95401", "95403", "95404", "95405", "95407", "95409"]
OCC_OPTIONS = [
    "Outdoor worker",
    "Indoor on-site",
    "Remote worker",
    "Retired",
    "Unemployed/Not in labor force",
]
VULN_OPTIONS = ["Low", "Medium", "High"]

BASE_CONNECTION_ARCHETYPES = [
    "Smartphone + internet",
    "Smartphone (limited data)",
    "Landline only",
    "No reliable phone/internet",
]

CHANNEL_OPTIONS = [
    "SMS text",
    "Robocall (voice call)",
    "Social media post",
    "Community meeting announcement",
    "Door-to-door outreach",
    "Printed flyer/poster",
    "Local radio/TV announcement",
]

MESSAGE_LANG_OPTIONS = ["English", "Spanish", "Vietnamese", "Tagalog", "Multilingual"]

DEFAULT_CONFIG = {
    "age_pct": {"0-17": 20, "18-39": 29, "40-64": 33, "65+": 18},
    "language_pct": {"English": 72, "Spanish": 20, "Vietnamese": 3, "Tagalog": 2, "Other": 3},
    "zip_pct": {"95401": 18, "95403": 24, "95404": 14, "95405": 10, "95407": 22, "95409": 12},
    "disability_yes_pct": 12.0,
}

DEFAULT_CONNECTION_LIBRARY = {
    "Smartphone + internet": {"base": "Smartphone + internet", "pct": 68.0},
    "Smartphone (limited data)": {"base": "Smartphone (limited data)", "pct": 16.0},
    "Landline only": {"base": "Landline only", "pct": 10.0},
    "No reliable phone/internet": {"base": "No reliable phone/internet", "pct": 6.0},
}


# -----------------------------
# Helpers
# -----------------------------
def safe_key(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", text)


def do_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def counts_from_percentages(pct_dict, n=100):
    keys = list(pct_dict.keys())
    vals = np.array([max(0.0, float(pct_dict[k])) for k in keys], dtype=float)
    if vals.sum() == 0:
        vals = np.ones_like(vals)

    norm = vals / vals.sum()
    raw = norm * n
    flo = np.floor(raw).astype(int)
    remainder = n - int(flo.sum())

    frac = raw - flo
    idx_sorted = np.argsort(-frac)
    for i in range(remainder):
        flo[idx_sorted[i]] += 1

    return {keys[i]: int(flo[i]) for i in range(len(keys))}


def expand_counts(count_dict):
    out = []
    for k, c in count_dict.items():
        out.extend([k] * int(c))
    return out


def sample_from_probs(rng, prob_dict):
    keys = list(prob_dict.keys())
    probs = np.array(list(prob_dict.values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, p=probs)


def edit_pct_dict(title, pct_dict, key_prefix):
    st.markdown(f"**{title}**")
    cols = st.columns(3)
    new_dict = {}
    items = list(pct_dict.items())
    for i, (k, v) in enumerate(items):
        with cols[i % 3]:
            new_val = st.number_input(
                f"{k}",
                min_value=0.0,
                max_value=100.0,
                value=float(v),
                step=0.5,
                key=f"{key_prefix}_{safe_key(k)}",
            )
            new_dict[k] = float(new_val)

    st.caption(f"Raw total entered: {sum(new_dict.values()):.1f}% (auto-normalized to 100% on generation)")
    return new_dict


def distribution_table(df, col, order=None):
    if order is None:
        counts = df[col].value_counts()
    else:
        counts = df[col].value_counts().reindex(order).fillna(0).astype(int)
    return pd.DataFrame(
        {
            col: counts.index,
            "Count (icons)": counts.values,
            "Percent": (counts.values / len(df) * 100).round(1),
        }
    )


# -----------------------------
# Population generation
# -----------------------------
def generate_population(config, connection_library, seed=SEED):
    rng = np.random.default_rng(seed)
    n = POP_SIZE

    age_counts = counts_from_percentages(config["age_pct"], n)
    lang_counts = counts_from_percentages(config["language_pct"], n)
    zip_counts = counts_from_percentages(config["zip_pct"], n)
    conn_pct = {k: v["pct"] for k, v in connection_library.items()}
    conn_counts = counts_from_percentages(conn_pct, n)

    ages = expand_counts(age_counts)
    langs = expand_counts(lang_counts)
    zips = expand_counts(zip_counts)
    conns = expand_counts(conn_counts)

    rng.shuffle(ages)
    rng.shuffle(langs)
    rng.shuffle(zips)
    rng.shuffle(conns)

    df = pd.DataFrame(
        {
            "resident_id": np.arange(1, n + 1),
            "age_group": ages,
            "primary_language": langs,
            "zip_code": zips,
            "connection_type": conns,
        }
    )
    df["connection_base"] = df["connection_type"].map(lambda x: connection_library[x]["base"])

    # English proficiency (approx)
    limited_prob_by_language = {
        "English": 0.01,
        "Spanish": 0.45,
        "Vietnamese": 0.50,
        "Tagalog": 0.30,
        "Other": 0.40,
    }
    df["english_proficiency"] = [
        "Limited" if rng.random() < limited_prob_by_language[lang] else "Proficient"
        for lang in df["primary_language"]
    ]

    # Disability assignment (user-set %, weighted toward older ages)
    target_yes = int(round((config["disability_yes_pct"] / 100.0) * n))
    target_yes = max(0, min(n, target_yes))
    age_weight = {"0-17": 0.7, "18-39": 0.7, "40-64": 1.2, "65+": 2.3}
    w = df["age_group"].map(age_weight).astype(float).values
    w = w / w.sum()
    yes_idx = rng.choice(df.index, size=target_yes, replace=False, p=w) if target_yes > 0 else []

    df["disability_status"] = "No"
    if target_yes > 0:
        df.loc[yes_idx, "disability_status"] = "Yes"

    # Occupation
    occupations = []
    for _, row in df.iterrows():
        age = row["age_group"]
        if age == "0-17":
            probs = {
                "Outdoor worker": 0.02,
                "Indoor on-site": 0.03,
                "Remote worker": 0.00,
                "Retired": 0.00,
                "Unemployed/Not in labor force": 0.95,
            }
        elif age == "18-39":
            probs = {
                "Outdoor worker": 0.20,
                "Indoor on-site": 0.42,
                "Remote worker": 0.20,
                "Retired": 0.00,
                "Unemployed/Not in labor force": 0.18,
            }
        elif age == "40-64":
            probs = {
                "Outdoor worker": 0.12,
                "Indoor on-site": 0.45,
                "Remote worker": 0.22,
                "Retired": 0.05,
                "Unemployed/Not in labor force": 0.16,
            }
        else:
            probs = {
                "Outdoor worker": 0.02,
                "Indoor on-site": 0.04,
                "Remote worker": 0.03,
                "Retired": 0.75,
                "Unemployed/Not in labor force": 0.16,
            }
        occupations.append(sample_from_probs(rng, probs))
    df["occupation"] = occupations

    # Housing type
    housing = []
    for _, row in df.iterrows():
        z = row["zip_code"]
        age = row["age_group"]

        probs = {"Single-family": 0.55, "Apartment": 0.32, "Mobile home": 0.10, "Unhoused/temporary": 0.03}
        if z == "95407":
            probs = {"Single-family": 0.42, "Apartment": 0.45, "Mobile home": 0.10, "Unhoused/temporary": 0.03}
        elif z == "95409":
            probs = {"Single-family": 0.66, "Apartment": 0.20, "Mobile home": 0.12, "Unhoused/temporary": 0.02}

        if age == "65+":
            probs["Mobile home"] += 0.03
            probs["Apartment"] -= 0.02
            probs["Single-family"] -= 0.01

        housing.append(sample_from_probs(rng, probs))
    df["housing_type"] = housing

    # Cooling access
    cooling = []
    for _, row in df.iterrows():
        h = row["housing_type"]
        if h == "Single-family":
            probs = {"Central AC": 0.58, "Window/portable AC": 0.22, "Fan only": 0.16, "No cooling": 0.04}
        elif h == "Apartment":
            probs = {"Central AC": 0.32, "Window/portable AC": 0.30, "Fan only": 0.28, "No cooling": 0.10}
        elif h == "Mobile home":
            probs = {"Central AC": 0.20, "Window/portable AC": 0.35, "Fan only": 0.32, "No cooling": 0.13}
        else:
            probs = {"Central AC": 0.00, "Window/portable AC": 0.00, "Fan only": 0.15, "No cooling": 0.85}
        cooling.append(sample_from_probs(rng, probs))
    df["cooling_access"] = cooling

    # Social connectedness
    connectedness = []
    for _, row in df.iterrows():
        age = row["age_group"]
        limited = row["english_proficiency"] == "Limited"
        disability = row["disability_status"] == "Yes"

        if age == "0-17":
            probs = {"Isolated": 0.10, "Moderately connected": 0.55, "Highly connected": 0.35}
        elif age == "65+":
            probs = {"Isolated": 0.30, "Moderately connected": 0.50, "Highly connected": 0.20}
        else:
            probs = {"Isolated": 0.20, "Moderately connected": 0.55, "Highly connected": 0.25}

        if limited:
            probs["Isolated"] += 0.08
            probs["Highly connected"] -= 0.05
            probs["Moderately connected"] -= 0.03

        if disability:
            probs["Isolated"] += 0.05
            probs["Highly connected"] -= 0.02
            probs["Moderately connected"] -= 0.03

        connectedness.append(sample_from_probs(rng, probs))
    df["social_connectedness"] = connectedness

    # Vulnerability score
    def vulnerability_score(row):
        score = 0
        if row["age_group"] == "65+":
            score += 2
        elif row["age_group"] == "0-17":
            score += 1
        if row["disability_status"] == "Yes":
            score += 2
        if row["cooling_access"] == "Fan only":
            score += 1
        elif row["cooling_access"] == "No cooling":
            score += 2
        if row["english_proficiency"] == "Limited":
            score += 1
        if row["social_connectedness"] == "Isolated":
            score += 1
        if row["occupation"] == "Outdoor worker":
            score += 1
        if row["housing_type"] in ["Mobile home", "Unhoused/temporary"]:
            score += 1
        if row["connection_base"] in ["Landline only", "No reliable phone/internet"]:
            score += 1
        return score

    df["vulnerability_score"] = df.apply(vulnerability_score, axis=1)
    df["vulnerability_level"] = df["vulnerability_score"].apply(
        lambda s: "High" if s >= 6 else ("Medium" if s >= 3 else "Low")
    )

    # Grid coordinates
    df["x"] = ((df["resident_id"] - 1) % 10) + 1
    df["y"] = 10 - ((df["resident_id"] - 1) // 10)

    # Contact tracking
    df["contact_count"] = 0

    return df


# -----------------------------
# Communication logic
# -----------------------------
def channel_reach_prob(row, channel):
    tech = row["connection_base"]
    soc = row["social_connectedness"]
    housing = row["housing_type"]
    age = row["age_group"]
    disability = row["disability_status"]

    if channel == "SMS text":
        if tech == "Smartphone + internet":
            return 0.95
        if tech == "Smartphone (limited data)":
            return 0.75
        return 0.0

    if channel == "Robocall (voice call)":
        if tech == "Smartphone + internet":
            return 0.80
        if tech == "Smartphone (limited data)":
            return 0.75
        if tech == "Landline only":
            return 0.85
        return 0.0

    if channel == "Social media post":
        if tech == "Smartphone + internet":
            return 0.85
        if tech == "Smartphone (limited data)":
            return 0.50
        return 0.0

    if channel == "Community meeting announcement":
        base = {"Highly connected": 0.80, "Moderately connected": 0.55, "Isolated": 0.25}[soc]
        if age == "0-17":
            base *= 0.70
        if disability == "Yes":
            base *= 0.85
        return float(np.clip(base, 0, 1))

    if channel == "Door-to-door outreach":
        if housing == "Unhoused/temporary":
            return 0.45
        base = 0.90
        if soc == "Isolated":
            base += 0.05
        return float(np.clip(base, 0, 1))

    if channel == "Printed flyer/poster":
        base = 0.30 if housing == "Unhoused/temporary" else 0.65
        if soc == "Highly connected":
            base += 0.08
        return float(np.clip(base, 0, 1))

    if channel == "Local radio/TV announcement":
        base = 0.65
        if age == "65+":
            base += 0.10
        if tech == "No reliable phone/internet":
            base -= 0.15
        return float(np.clip(base, 0, 1))

    return 0.0


def language_match_prob(row, message_language):
    primary = row["primary_language"]
    eng_prof = row["english_proficiency"]

    if message_language == "Multilingual":
        return 1.0
    if message_language == primary:
        return 0.98
    if message_language == "English":
        return 0.80 if eng_prof == "Proficient" else 0.20
    return 0.15 if eng_prof == "Proficient" else 0.03


def contact_status(c):
    if c == 0:
        return "Never contacted"
    if c == 1:
        return "Contacted once"
    if c == 2:
        return "Contacted twice"
    return "Contacted 3+ times"


def apply_tactic(pop_df, rng, channel, message_language, intended_real_reach, filters, fractional_bank):
    mask = (
        pop_df["age_group"].isin(filters["age_groups"])
        & pop_df["zip_code"].isin(filters["zip_codes"])
        & pop_df["occupation"].isin(filters["occupations"])
        & pop_df["vulnerability_level"].isin(filters["vulnerability_levels"])
        & pop_df["primary_language"].isin(filters["primary_languages"])
        & pop_df["disability_status"].isin(filters["disability_statuses"])
        & pop_df["connection_type"].isin(filters["connection_types"])
    )

    targeted_count = int(mask.sum())
    if targeted_count == 0:
        return 0, 0, targeted_count, 0, fractional_bank

    weights = np.zeros(len(pop_df), dtype=float)
    for idx, row in pop_df[mask].iterrows():
        weights[idx] = channel_reach_prob(row, channel) * language_match_prob(row, message_language)

    eligible_idx = np.where(weights > 0)[0]
    eligible_count = int(len(eligible_idx))
    if eligible_count == 0:
        return 0, 0, targeted_count, eligible_count, fractional_bank

    expected_icons = (intended_real_reach / REP_FACTOR) + fractional_bank
    to_contact = int(np.floor(expected_icons))
    new_bank = expected_icons - to_contact

    if to_contact <= 0:
        return 0, 0, targeted_count, eligible_count, new_bank

    to_contact = min(to_contact, eligible_count)
    probs = weights[eligible_idx]
    probs = probs / probs.sum()

    selected = rng.choice(eligible_idx, size=to_contact, replace=False, p=probs)
    pop_df.loc[selected, "contact_count"] = pop_df.loc[selected, "contact_count"] + 1

    return int(to_contact), int(to_contact * REP_FACTOR), targeted_count, eligible_count, new_bank


# -----------------------------
# App
# -----------------------------
def init_state():
    if "config" not in st.session_state:
        st.session_state.config = deepcopy(DEFAULT_CONFIG)
    if "connection_library" not in st.session_state:
        st.session_state.connection_library = deepcopy(DEFAULT_CONNECTION_LIBRARY)
    if "population" not in st.session_state:
        st.session_state.population = generate_population(
            st.session_state.config, st.session_state.connection_library, seed=SEED
        )
    if "history" not in st.session_state:
        st.session_state.history = []
    if "rng" not in st.session_state:
        st.session_state.rng = np.random.default_rng(SEED)
    if "fractional_bank" not in st.session_state:
        st.session_state.fractional_bank = 0.0
    if "population_locked" not in st.session_state:
        st.session_state.population_locked = False


def main():
    st.set_page_config(page_title="Santa Rosa Heat Communication Simulation", layout="wide")
    init_state()

    st.title("Santa Rosa Extreme Heat Communication Simulation")
    st.info("Each icon = ~1,800 residents. Synthetic population size = 100.")

    tabs = st.tabs(["1) Population Setup", "2) Run Communication Tactics", "3) Data Sources"])

    # -----------------------------
    # Tab 1: Population Setup
    # -----------------------------
    with tabs[0]:
        st.subheader("Add / Edit Population Profile")
        st.caption("Enter percentages. They do not need to sum to 100; the app auto-normalizes.")

        st.session_state.config["age_pct"] = edit_pct_dict(
            "Age distribution (%)", st.session_state.config["age_pct"], "agepct"
        )
        st.session_state.config["language_pct"] = edit_pct_dict(
            "Primary language distribution (%)", st.session_state.config["language_pct"], "langpct"
        )
        st.session_state.config["zip_pct"] = edit_pct_dict(
            "ZIP distribution (%)", st.session_state.config["zip_pct"], "zippct"
        )

        st.session_state.config["disability_yes_pct"] = float(
            st.number_input(
                "Disability prevalence (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.config["disability_yes_pct"]),
                step=0.5,
                key="disability_yes_pct",
            )
        )

        st.markdown("---")
        st.subheader("Add Connection Type Section")
        st.caption("You can add custom connection types and map them to a base delivery behavior.")

        lib = st.session_state.connection_library
        for label in list(lib.keys()):
            sk = safe_key(label)
            c1, c2 = st.columns([1.1, 1.4])
            with c1:
                pct_val = st.number_input(
                    f"{label} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(lib[label]["pct"]),
                    step=0.5,
                    key=f"connpct_{sk}",
                )
            with c2:
                base_idx = BASE_CONNECTION_ARCHETYPES.index(lib[label]["base"])
                base_val = st.selectbox(
                    f"{label} behavior map",
                    BASE_CONNECTION_ARCHETYPES,
                    index=base_idx,
                    key=f"connbase_{sk}",
                )
            lib[label]["pct"] = float(pct_val)
            lib[label]["base"] = base_val

        st.caption(
            f"Raw connection total entered: {sum([v['pct'] for v in lib.values()]):.1f}% "
            "(auto-normalized to 100% on generation)"
        )

        with st.expander("➕ Add a new custom connection type"):
            new_name = st.text_input("Connection type name", placeholder="e.g., Family phone tree")
            new_base = st.selectbox("Map behavior to", BASE_CONNECTION_ARCHETYPES, key="new_conn_base")
            new_pct = st.number_input("Initial %", 0.0, 100.0, 2.0, 0.5, key="new_conn_pct")
            if st.button("Add connection type"):
                nm = new_name.strip()
                if nm == "":
                    st.warning("Please enter a name.")
                elif nm in lib:
                    st.warning("That connection type already exists.")
                else:
                    lib[nm] = {"base": new_base, "pct": float(new_pct)}
                    st.success(f"Added '{nm}'.")
                    do_rerun()

        custom_types = [k for k in lib.keys() if k not in BASE_CONNECTION_ARCHETYPES]
        with st.expander("🗑 Remove custom connection type"):
            if len(custom_types) == 0:
                st.caption("No custom connection types yet.")
            else:
                rm = st.selectbox("Select custom type to remove", custom_types)
                if st.button("Remove selected connection type"):
                    del lib[rm]
                    st.success(f"Removed '{rm}'.")
                    do_rerun()

        st.markdown("---")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Generate / Regenerate population", type="primary"):
                st.session_state.population = generate_population(
                    st.session_state.config, st.session_state.connection_library, seed=SEED
                )
                st.session_state.history = []
                st.session_state.fractional_bank = 0.0
                st.session_state.rng = np.random.default_rng(SEED)
                st.session_state.population_locked = False
                st.success("Population generated. Review below, then lock to begin tactics.")
        with b2:
            if st.button("Lock population & start tactics"):
                st.session_state.population_locked = True
                st.success("Population locked. Go to 'Run Communication Tactics'.")
        with b3:
            if st.button("Unlock population for editing"):
                st.session_state.population_locked = False
                st.warning("Population unlocked. Re-lock before running tactics.")

        lock_msg = "LOCKED ✅" if st.session_state.population_locked else "UNLOCKED ⚠️"
        st.write(f"Current status: **{lock_msg}**")

        st.subheader("Population Preview")
        p = st.session_state.population
        c1, c2 = st.columns(2)
        with c1:
            st.write("Age")
            st.dataframe(distribution_table(p, "age_group", AGE_OPTIONS), use_container_width=True, hide_index=True)
            st.write("Language")
            st.dataframe(distribution_table(p, "primary_language", LANG_OPTIONS), use_container_width=True, hide_index=True)
            st.write("Disability")
            st.dataframe(distribution_table(p, "disability_status", ["Yes", "No"]), use_container_width=True, hide_index=True)
        with c2:
            st.write("ZIP")
            st.dataframe(distribution_table(p, "zip_code", ZIP_OPTIONS), use_container_width=True, hide_index=True)
            st.write("Connection type")
            st.dataframe(
                distribution_table(p, "connection_type", list(st.session_state.connection_library.keys())),
                use_container_width=True,
                hide_index=True,
            )

    # -----------------------------
    # Tab 2: Tactics
    # -----------------------------
    with tabs[1]:
        pop_df = st.session_state.population
        pop_df["contact_status"] = pop_df["contact_count"].apply(contact_status)

        if not st.session_state.population_locked:
            st.warning("Population is unlocked. Lock it in Tab 1 before applying communication tactics.")

        left, right = st.columns([1.05, 1.95])

        with left:
            st.subheader("Tactic Builder")
            channel = st.selectbox("Channel", CHANNEL_OPTIONS)
            message_language = st.selectbox("Message language", MESSAGE_LANG_OPTIONS)
            intended_reach = st.slider(
                "Intended reach (real residents)",
                min_value=100,
                max_value=50000,
                value=1800,
                step=100,
            )
            st.caption(f"Equivalent to {intended_reach / REP_FACTOR:.2f} icons")

            st.markdown("**Targeting filters**")
            age_filter = st.multiselect("Age groups", AGE_OPTIONS, default=AGE_OPTIONS)
            zip_filter = st.multiselect("ZIP codes", ZIP_OPTIONS, default=ZIP_OPTIONS)
            occ_filter = st.multiselect("Occupation", OCC_OPTIONS, default=OCC_OPTIONS)
            vuln_filter = st.multiselect("Vulnerability level", VULN_OPTIONS, default=VULN_OPTIONS)
            lang_filter = st.multiselect("Primary language", LANG_OPTIONS, default=LANG_OPTIONS)
            dis_filter = st.multiselect("Disability status", ["Yes", "No"], default=["Yes", "No"])
            conn_options = sorted(pop_df["connection_type"].unique().tolist())
            conn_filter = st.multiselect("Connection type", conn_options, default=conn_options)

            apply_btn = st.button("Apply tactic", type="primary", disabled=not st.session_state.population_locked)
            reset_btn = st.button("Reset contacts (keep population)")

            if reset_btn:
                st.session_state.population["contact_count"] = 0
                st.session_state.history = []
                st.session_state.fractional_bank = 0.0
                st.session_state.rng = np.random.default_rng(SEED)
                st.success("Contacts reset. Population unchanged.")
                do_rerun()

            if apply_btn:
                if any(len(x) == 0 for x in [age_filter, zip_filter, occ_filter, vuln_filter, lang_filter, dis_filter, conn_filter]):
                    st.warning("Each filter must have at least one selected value.")
                else:
                    filters = {
                        "age_groups": age_filter,
                        "zip_codes": zip_filter,
                        "occupations": occ_filter,
                        "vulnerability_levels": vuln_filter,
                        "primary_languages": lang_filter,
                        "disability_statuses": dis_filter,
                        "connection_types": conn_filter,
                    }

                    contacted_icons, contacted_real, targeted_count, eligible_count, new_bank = apply_tactic(
                        st.session_state.population,
                        st.session_state.rng,
                        channel,
                        message_language,
                        intended_reach,
                        filters,
                        st.session_state.fractional_bank,
                    )
                    st.session_state.fractional_bank = new_bank

                    reached_icons_total = int((st.session_state.population["contact_count"] > 0).sum())
                    reached_pct_total = reached_icons_total / POP_SIZE * 100

                    st.session_state.history.append(
                        {
                            "Step": len(st.session_state.history) + 1,
                            "Channel": channel,
                            "Language": message_language,
                            "Intended reach (real)": intended_reach,
                            "Targeted icons": targeted_count,
                            "Eligible icons": eligible_count,
                            "Contacted this step (icons)": contacted_icons,
                            "Contacted this step (real approx)": contacted_real,
                            "Cumulative reached %": round(reached_pct_total, 1),
                        }
                    )

                    if contacted_icons == 0:
                        st.warning("No new icon-level contacts this step (small reach or narrow eligibility).")
                    else:
                        st.success(f"Applied: {contacted_icons} icons contacted (~{contacted_real:,} residents).")
                    do_rerun()

        with right:
            p2 = st.session_state.population.copy()
            p2["contact_status"] = p2["contact_count"].apply(contact_status)

            vega_spec = {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "height": 620,
                "mark": {"type": "point", "filled": True, "size": 220, "stroke": "black", "strokeWidth": 0.6},
                "encoding": {
                    "x": {"field": "x", "type": "quantitative", "axis": None, "scale": {"domain": [0.5, 10.5]}},
                    "y": {"field": "y", "type": "quantitative", "axis": None, "scale": {"domain": [0.5, 10.5]}},
                    "color": {
                        "field": "contact_status",
                        "type": "nominal",
                        "scale": {
                            "domain": ["Never contacted", "Contacted once", "Contacted twice", "Contacted 3+ times"],
                            "range": ["#cbd5e1", "#60a5fa", "#f59e0b", "#ef4444"],
                        },
                        "legend": {"title": "Contact frequency"},
                    },
                    "shape": {
                        "field": "vulnerability_level",
                        "type": "nominal",
                        "scale": {"domain": ["Low", "Medium", "High"], "range": ["circle", "square", "diamond"]},
                        "legend": {"title": "Heat vulnerability"},
                    },
                    "tooltip": [
                        {"field": "resident_id", "type": "quantitative", "title": "Resident ID"},
                        {"field": "age_group", "type": "nominal", "title": "Age"},
                        {"field": "primary_language", "type": "nominal", "title": "Language"},
                        {"field": "english_proficiency", "type": "nominal", "title": "English proficiency"},
                        {"field": "disability_status", "type": "nominal", "title": "Disability"},
                        {"field": "zip_code", "type": "nominal", "title": "ZIP"},
                        {"field": "occupation", "type": "nominal", "title": "Occupation"},
                        {"field": "connection_type", "type": "nominal", "title": "Connection type"},
                        {"field": "connection_base", "type": "nominal", "title": "Connection base"},
                        {"field": "cooling_access", "type": "nominal", "title": "Cooling"},
                        {"field": "social_connectedness", "type": "nominal", "title": "Connectedness"},
                        {"field": "contact_count", "type": "quantitative", "title": "Contact count"},
                    ],
                },
                "config": {"view": {"stroke": None}},
            }

            st.subheader("Synthetic Population Grid")
            st.caption("Color = contact frequency | Shape = vulnerability")
            st.vega_lite_chart(p2, vega_spec, use_container_width=True)

            reached_icons = int((p2["contact_count"] > 0).sum())
            never_icons = POP_SIZE - reached_icons
            reached_pct = reached_icons / POP_SIZE * 100

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Population reached", f"{reached_pct:.1f}%")
            c2.metric("Reached (real approx)", f"{reached_icons * REP_FACTOR:,}")
            c3.metric("Never contacted icons", f"{never_icons}")
            c4.metric("Avg contacts/icon", f"{p2['contact_count'].mean():.2f}")

            st.markdown("### Who is still being missed? (never-contacted)")
            missed = p2[p2["contact_count"] == 0].copy()
            if len(missed) == 0:
                st.success("All icons contacted at least once.")
            else:
                breakdown = {
                    "Age 65+": int((missed["age_group"] == "65+").sum()),
                    "Limited English proficiency": int((missed["english_proficiency"] == "Limited").sum()),
                    "Disability = Yes": int((missed["disability_status"] == "Yes").sum()),
                    "Low/no cooling": int(missed["cooling_access"].isin(["Fan only", "No cooling"]).sum()),
                    "Isolated": int((missed["social_connectedness"] == "Isolated").sum()),
                    "Outdoor workers": int((missed["occupation"] == "Outdoor worker").sum()),
                    "Low connectivity (landline/none)": int(
                        missed["connection_base"].isin(["Landline only", "No reliable phone/internet"]).sum()
                    ),
                }
                bdf = pd.DataFrame(
                    {
                        "Trait among never-contacted": list(breakdown.keys()),
                        "Count (icons)": list(breakdown.values()),
                    }
                ).sort_values("Count (icons)", ascending=False)
                bdf["Approx real residents"] = bdf["Count (icons)"] * REP_FACTOR
                st.dataframe(bdf, use_container_width=True, hide_index=True)

            st.markdown("### Tactic history")
            if len(st.session_state.history) == 0:
                st.write("No tactics applied yet.")
            else:
                st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

    # -----------------------------
    # Tab 3: Data Sources
    # -----------------------------
    with tabs[2]:
        st.subheader("Data Sources & Assumptions")

        source_rows = [
            ["Total population scaling", "Census city population estimate / ACS profile", "Set to 180,000 for scenario scaling (100 icons x 1,800)."],
            ["Age distribution", "ACS S0101 / DP05", "Used as editable default percentages."],
            ["Primary language", "ACS S1601 / B16001", "Used as editable default percentages."],
            ["English proficiency", "ACS S1601 / B16004", "Applied via language-specific LEP probabilities."],
            ["Disability prevalence", "ACS S1810", "Single editable percentage, then assigned with age-weighting."],
            ["ZIP distribution", "ACS ZCTA pop (e.g., B01003) + local crosswalk method", "Used as editable default ZIP mix."],
            ["Occupation", "ACS S2401 + modeled mapping", "Mapped into exercise categories (outdoor, remote, retired, etc.)."],
            ["Housing type", "ACS B25024 / S2504 + modeled assumptions", "Generated by ZIP/age conditional distributions."],
            ["Cooling access", "Modeled (proxy-based)", "Not directly a single ACS field; proxied by housing context."],
            ["Connection type", "ACS S2801 / B28002 + modeled mapping", "Editable section; channel behavior maps to 4 base archetypes."],
            ["Social connectedness", "Modeled proxy", "Proxy logic using age, LEP, disability."],
        ]
        sdf = pd.DataFrame(source_rows, columns=["Variable", "Source (ACS/Census)", "How used in app"])
        st.dataframe(sdf, use_container_width=True, hide_index=True)

        st.markdown(
            """
**Notes**
- This tool uses a **synthetic population** for tabletop planning.
- Core demographic defaults align to ACS/Census-style profiles, but some fields are **modeled assumptions**.
- All data are generated inside code (no external files needed for deployment).
"""
        )


if __name__ == "__main__":
    main()
