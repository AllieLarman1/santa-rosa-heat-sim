import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Constants
# -----------------------------
POP_SIZE = 100
REP_FACTOR = 1800  # each simulated resident represents ~1,800 real residents
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


def allocate_counts_from_percent(percent_dict, n=100):
    raw = {k: (v / 100.0) * n for k, v in percent_dict.items()}
    floored = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = n - sum(floored.values())

    frac_parts = sorted(
        [(k, raw[k] - floored[k]) for k in raw.keys()],
        key=lambda x: x[1],
        reverse=True,
    )
    for i in range(remainder):
        floored[frac_parts[i][0]] += 1
    return floored


def expand_counts(count_dict):
    vals = []
    for k, c in count_dict.items():
        vals.extend([k] * c)
    return vals


def sample_from_probs(rng, prob_dict):
    keys = list(prob_dict.keys())
    probs = np.array(list(prob_dict.values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, p=probs)


@st.cache_data
def generate_population(seed=SEED):
    rng = np.random.default_rng(seed)
    n = POP_SIZE

    age_counts = allocate_counts_from_percent(
        {"0-17": 20, "18-39": 29, "40-64": 33, "65+": 18}, n
    )
    lang_counts = allocate_counts_from_percent(
        {"English": 72, "Spanish": 20, "Vietnamese": 3, "Tagalog": 2, "Other": 3}, n
    )
    zip_counts = allocate_counts_from_percent(
        {"95401": 18, "95403": 24, "95404": 14, "95405": 10, "95407": 22, "95409": 12}, n
    )

    ages = expand_counts(age_counts)
    langs = expand_counts(lang_counts)
    zips = expand_counts(zip_counts)

    rng.shuffle(ages)
    rng.shuffle(langs)
    rng.shuffle(zips)

    df = pd.DataFrame(
        {
            "resident_id": np.arange(1, n + 1),
            "age_group": ages,
            "primary_language": langs,
            "zip_code": zips,
        }
    )

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

    disability_target = 12
    age_weight = {"0-17": 0.7, "18-39": 0.7, "40-64": 1.2, "65+": 2.3}
    w = df["age_group"].map(age_weight).astype(float).values
    w = w / w.sum()
    yes_idx = rng.choice(df.index, size=disability_target, replace=False, p=w)

    df["disability_status"] = "No"
    df.loc[yes_idx, "disability_status"] = "Yes"

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

    housing = []
    for _, row in df.iterrows():
        z = row["zip_code"]
        age = row["age_group"]

        probs = {
            "Single-family": 0.55,
            "Apartment": 0.32,
            "Mobile home": 0.10,
            "Unhoused/temporary": 0.03,
        }

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

    tech = []
    for _, row in df.iterrows():
        age = row["age_group"]
        h = row["housing_type"]

        if age == "0-17":
            probs = {
                "Smartphone + internet": 0.78,
                "Smartphone (limited data)": 0.16,
                "Landline only": 0.04,
                "No reliable phone/internet": 0.02,
            }
        elif age == "18-39":
            probs = {
                "Smartphone + internet": 0.82,
                "Smartphone (limited data)": 0.12,
                "Landline only": 0.03,
                "No reliable phone/internet": 0.03,
            }
        elif age == "40-64":
            probs = {
                "Smartphone + internet": 0.70,
                "Smartphone (limited data)": 0.15,
                "Landline only": 0.10,
                "No reliable phone/internet": 0.05,
            }
        else:
            probs = {
                "Smartphone + internet": 0.45,
                "Smartphone (limited data)": 0.14,
                "Landline only": 0.30,
                "No reliable phone/internet": 0.11,
            }

        if h == "Unhoused/temporary":
            probs = {
                "Smartphone + internet": 0.20,
                "Smartphone (limited data)": 0.35,
                "Landline only": 0.05,
                "No reliable phone/internet": 0.40,
            }

        tech.append(sample_from_probs(rng, probs))
    df["tech_access"] = tech

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
        if row["tech_access"] in ["Landline only", "No reliable phone/internet"]:
            score += 1
        return score

    df["vulnerability_score"] = df.apply(vulnerability_score, axis=1)
    df["vulnerability_level"] = df["vulnerability_score"].apply(
        lambda s: "High" if s >= 6 else ("Medium" if s >= 3 else "Low")
    )

    df["x"] = ((df["resident_id"] - 1) % 10) + 1
    df["y"] = 10 - ((df["resident_id"] - 1) // 10)
    df["contact_count"] = 0
    return df


def channel_reach_prob(row, channel):
    tech = row["tech_access"]
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
        if housing == "Unhoused/temporary":
            base = 0.30
        else:
            base = 0.65
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


def distribution_table(df, col, order=None):
    if order is None:
        counts = df[col].value_counts()
    else:
        counts = df[col].value_counts().reindex(order).fillna(0).astype(int)
    out = pd.DataFrame(
        {
            col: counts.index,
            "Count (sim residents)": counts.values,
            "Percent": (counts.values / len(df) * 100).round(1),
        }
    )
    return out


def main():
    st.set_page_config(page_title="Santa Rosa Heat Communication Simulation", layout="wide")
    st.title("Santa Rosa Extreme Heat Communication Simulation")
    st.caption("Live tabletop exercise tool for communication strategy testing")
    st.info("Each icon = ~1,800 residents. Synthetic population size = 100.")

    if "population" not in st.session_state:
        st.session_state.population = generate_population().copy(deep=True)
    if "history" not in st.session_state:
        st.session_state.history = []
    if "rng" not in st.session_state:
        st.session_state.rng = np.random.default_rng(SEED)
    if "fractional_bank" not in st.session_state:
        st.session_state.fractional_bank = 0.0

    pop_df = st.session_state.population

    with st.sidebar:
        st.header("Tactic Builder")
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

        st.subheader("Optional targeting filters")
        age_filter = st.multiselect("Age groups", AGE_OPTIONS, default=AGE_OPTIONS)
        zip_filter = st.multiselect("ZIP codes", ZIP_OPTIONS, default=ZIP_OPTIONS)
        occ_filter = st.multiselect("Occupation", OCC_OPTIONS, default=OCC_OPTIONS)
        vuln_filter = st.multiselect("Vulnerability level", VULN_OPTIONS, default=VULN_OPTIONS)
        lang_filter = st.multiselect("Primary language", LANG_OPTIONS, default=LANG_OPTIONS)
        dis_filter = st.multiselect("Disability status", ["Yes", "No"], default=["Yes", "No"])

        apply_btn = st.button("Apply tactic", type="primary")
        reset_btn = st.button("Reset scenario")

    if reset_btn:
        st.session_state.population = generate_population().copy(deep=True)
        st.session_state.history = []
        st.session_state.rng = np.random.default_rng(SEED)
        st.session_state.fractional_bank = 0.0
        st.success("Scenario reset complete.")
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    if apply_btn:
        if any(len(x) == 0 for x in [age_filter, zip_filter, occ_filter, vuln_filter, lang_filter, dis_filter]):
            st.warning("At least one value must be selected in each filter.")
        else:
            filters = {
                "age_groups": age_filter,
                "zip_codes": zip_filter,
                "occupations": occ_filter,
                "vulnerability_levels": vuln_filter,
                "primary_languages": lang_filter,
                "disability_statuses": dis_filter,
            }

            contacted_icons, contacted_real, targeted_count, eligible_count, new_bank = apply_tactic(
                pop_df,
                st.session_state.rng,
                channel,
                message_language,
                intended_reach,
                filters,
                st.session_state.fractional_bank,
            )
            st.session_state.fractional_bank = new_bank

            reached_icons_total = int((pop_df["contact_count"] > 0).sum())
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
                st.success(f"Tactic applied: {contacted_icons} icons contacted (~{contacted_real:,} residents).")

    pop_df["contact_status"] = pop_df["contact_count"].apply(contact_status)

    # Vega-Lite chart (no extra plotting library dependency)
    vega_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "height": 620,
        "mark": {"type": "point", "filled": True, "size": 220, "stroke": "black", "strokeWidth": 0.6},
        "encoding": {
            "x": {
                "field": "x",
                "type": "quantitative",
                "axis": None,
                "scale": {"domain": [0.5, 10.5]},
            },
            "y": {
                "field": "y",
                "type": "quantitative",
                "axis": None,
                "scale": {"domain": [0.5, 10.5]},
            },
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
                {"field": "primary_language", "type": "nominal", "title": "Primary language"},
                {"field": "english_proficiency", "type": "nominal", "title": "English proficiency"},
                {"field": "disability_status", "type": "nominal", "title": "Disability"},
                {"field": "zip_code", "type": "nominal", "title": "ZIP"},
                {"field": "occupation", "type": "nominal", "title": "Occupation"},
                {"field": "housing_type", "type": "nominal", "title": "Housing"},
                {"field": "cooling_access", "type": "nominal", "title": "Cooling"},
                {"field": "tech_access", "type": "nominal", "title": "Tech access"},
                {"field": "social_connectedness", "type": "nominal", "title": "Connectedness"},
                {"field": "contact_count", "type": "quantitative", "title": "Contact count"},
            ],
        },
        "config": {"view": {"stroke": None}},
    }

    st.subheader("Synthetic Santa Rosa Population (100 icons)")
    st.caption("Color = contact frequency | Shape = heat vulnerability")
    st.vega_lite_chart(pop_df, vega_spec, use_container_width=True)

    reached_icons = int((pop_df["contact_count"] > 0).sum())
    never_icons = POP_SIZE - reached_icons
    reached_pct = reached_icons / POP_SIZE * 100
    avg_contacts = pop_df["contact_count"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Population reached", f"{reached_pct:.1f}%")
    c2.metric("Reached (real approx)", f"{reached_icons * REP_FACTOR:,}")
    c3.metric("Never contacted icons", f"{never_icons}")
    c4.metric("Avg contacts/icon", f"{avg_contacts:.2f}")

    st.subheader("Who is still being missed? (never-contacted icons)")
    missed = pop_df[pop_df["contact_count"] == 0].copy()
    if len(missed) == 0:
        st.success("All simulated residents have been contacted at least once.")
    else:
        breakdown = {
            "Age 65+": (missed["age_group"] == "65+").sum(),
            "Limited English proficiency": (missed["english_proficiency"] == "Limited").sum(),
            "Disability status = Yes": (missed["disability_status"] == "Yes").sum(),
            "Low/no cooling (fan only or no cooling)": missed["cooling_access"].isin(["Fan only", "No cooling"]).sum(),
            "Socially isolated": (missed["social_connectedness"] == "Isolated").sum(),
            "Outdoor workers": (missed["occupation"] == "Outdoor worker").sum(),
            "Low tech access (landline only/no reliable)": missed["tech_access"].isin(
                ["Landline only", "No reliable phone/internet"]
            ).sum(),
        }

        bdf = pd.DataFrame(
            {
                "Vulnerability trait among never-contacted": list(breakdown.keys()),
                "Count (icons)": list(breakdown.values()),
            }
        )
        bdf["Approx real residents"] = bdf["Count (icons)"] * REP_FACTOR
        bdf = bdf.sort_values("Count (icons)", ascending=False)
        st.dataframe(bdf, use_container_width=True, hide_index=True)

    st.subheader("Tactic history")
    if len(st.session_state.history) == 0:
        st.write("No tactics applied yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

    with st.expander("Assumptions and data notes"):
        st.markdown(
            """
- Synthetic population of 100 icons; each icon ≈ 1,800 real residents.
- Fixed random seed for reproducibility.
- Approximate ACS/Census profile proportions used for realism in exercise context.
- Planning/exercise tool; not an operational forecast model.
"""
        )

    with st.expander("Quick check of synthetic composition"):
        st.write("Age")
        st.dataframe(distribution_table(pop_df, "age_group", AGE_OPTIONS), use_container_width=True, hide_index=True)
        st.write("Primary language")
        st.dataframe(distribution_table(pop_df, "primary_language", LANG_OPTIONS), use_container_width=True, hide_index=True)
        st.write("Disability status")
        st.dataframe(distribution_table(pop_df, "disability_status", ["Yes", "No"]), use_container_width=True, hide_index=True)
        st.write("ZIP code")
        st.dataframe(distribution_table(pop_df, "zip_code", ZIP_OPTIONS), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
