import pickle

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
from scipy.stats import spearmanr, entropy
from scipy.spatial.distance import jensenshannon

from age_of_acquisition import calc_ages_of_acquisition, COMPREHENSION_SPEECH_ACTS_ENOUGH_DATA_2_OCCURRENCES, MAX_AGE
from utils import age_bin, calculate_frequencies, SOURCE_CRF, SOURCE_SNOW, TARGET_PRODUCTION, TARGET_COMPREHENSION, \
    AGES, load_whole_childes_data, SPEECH_ACT, CHILD, PATH_NEW_ENGLAND_UTTERANCES_ANNOTATED, AGES_LONG

SOURCE_SNOW_LABEL = "Data from Snow et al. (1996)"
SOURCE_AUTOMATIC_NEW_ENGLAND_LABEL = "Automatically Annotated Data (New England)"
SOURCE_AUTOMATIC_CHILDES_LABEL = "Automatically Annotated Data (English CHILDES)"
ORDER = [SOURCE_SNOW_LABEL, SOURCE_AUTOMATIC_NEW_ENGLAND_LABEL, SOURCE_AUTOMATIC_CHILDES_LABEL]

MAX_NUM_SPEECH_ACT_TYPES = 25

AGE_OF_ACQUISITION_MIN_DATAPOINTS = 2
AGE_OF_ACQUISITION_SPEECH_ACTS_ENOUGH_DATA = [
    "RP",
    "QN",
    "TO",
    "ST",
    "RR",
    "SA",
    "AP",
    "AC",
    "YQ",
    "MK",
    "CL",
    "PF",
    "SI",
    "RT",
    "PR",
    "RD",
    "FP",
    "AD",
    "CS",
    "DC",
    "AN",
    "PA",
    "DW",
    "AA",
    "SC",
]
AGE_OF_ACQUISITION_SPEECH_ACTS_ENOUGH_DATA_NO_DECLINE = [
    "RP",
    "QN",
    "ST",
    "RR",
    "SA",
    "AP",
    "AC",
    "YQ",
    "MK",
    "CL",
    "SI",
    "RT",
    "PR",
    "RD",
    "FP",
    "AD",
    "CS",
    "DC",
    "AN",
    "PA",
    "DW",
    "AA",
    "SC",
]


def calculate_num_speech_act_types(data, column_name_speech_act):
    # number of speech act types at different ages
    results = []
    for age in AGES:
        data_age = data[data["age_months"] == age]

        results_age = {}
        for num_speech_act_types in range(MAX_NUM_SPEECH_ACT_TYPES):
            results_age[num_speech_act_types] = 0

        children_ids = data_age.file_id.unique()
        for child_id in children_ids:
            speech_acts_child = data_age[
                (data_age.file_id == child_id)
                & (data_age.speaker == CHILD)
                & (~data_age[column_name_speech_act].isin(["YY", "OO"]))
            ][column_name_speech_act]
            speech_act_type_counts = speech_acts_child.value_counts().to_list()

            num_produced_speech_act_types = len(
                [c for c in speech_act_type_counts if c >= 2]
            )
            if num_produced_speech_act_types >= MAX_NUM_SPEECH_ACT_TYPES:
                num_produced_speech_act_types = MAX_NUM_SPEECH_ACT_TYPES - 1

            results_age[num_produced_speech_act_types] += 1

        # Normalize children counts to get proportions
        for num_speech_act_types in results_age.keys():
            results_age[num_speech_act_types] = results_age[num_speech_act_types] / len(
                children_ids
            )

        for num_speech_act_types in range(MAX_NUM_SPEECH_ACT_TYPES):
            results.append(
                {
                    "age": age,
                    "num_types": num_speech_act_types,
                    "frac_children": results_age[num_speech_act_types],
                }
            )

    return pd.DataFrame(results)


def reproduce_num_speech_acts(data, data_whole_childes):
    results_snow = calculate_num_speech_act_types(data, SPEECH_ACT)
    results_snow["source"] = SOURCE_SNOW_LABEL

    results_crf = calculate_num_speech_act_types(data, "y_pred")
    results_crf["source"] = SOURCE_AUTOMATIC_NEW_ENGLAND_LABEL

    results_childes = calculate_num_speech_act_types(data_whole_childes, "y_pred")
    results_childes["source"] = SOURCE_AUTOMATIC_CHILDES_LABEL

    results = results_snow.append(results_crf).append(results_childes)

    # Calculate KL divergences:
    for age in AGES:
        kl_divergence = entropy(
            results_crf[results_crf.age == age].frac_children.to_list(),
            qk=results_snow[results_snow.age == age].frac_children.to_list(),
        )
        print(f"KL Divergence (NewEngland, {age} months): {kl_divergence:.3f}")

        kl_divergence = entropy(
            results_childes[results_childes.age == age].frac_children.to_list(),
            qk=results_snow[results_snow.age == age].frac_children.to_list(),
        )
        print(f"KL Divergence (CHILDES, {age} months): {kl_divergence:.3f}")

        jensen_shannon_distance = jensenshannon(
            results_crf[results_crf.age == age].frac_children.to_list(),
            results_snow[results_snow.age == age].frac_children.to_list(),
        )
        print(
            f"Jensen-Shannon Distance (NewEngland, {age} months): {jensen_shannon_distance:.3f}"
        )

        jensen_shannon_distance = jensenshannon(
            results_childes[results_childes.age == age].frac_children.to_list(),
            results_snow[results_snow.age == age].frac_children.to_list(),
        )
        print(
            f"Jensen-Shannon Distance (CHILDES, {age} months): {jensen_shannon_distance:.3f}"
        )

    fig, (axes) = plt.subplots(3, 1, sharex="all")

    # Move title into figure
    plt.rcParams["axes.titley"] = 1.0
    plt.rcParams["axes.titlepad"] = -14

    for i, age in enumerate(AGES):
        results_age = results[results.age == age]

        sns.barplot(
            ax=axes[i],
            x="num_types",
            hue="source",
            y="frac_children",
            data=results_age,
            hue_order=ORDER,
        )

        if i == 0:
            axes[i].legend(loc="upper right", bbox_to_anchor=(1, 0.8))
        else:
            axes[i].legend_.remove()

        axes[i].set_title(f"Age: {age} months")

        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

        if age > 14:
            axes[i].set_ylim(0, 0.23)

    axes[1].set_ylabel("proportion of children")
    plt.xlabel("number of different speech acts produced")
    plt.tight_layout()
    plt.show()


def calculate_freq_distributions(
    data, column_name_speech_act, speech_acts_analyzed, age, source
):
    # number of speech act types at different ages
    data_age = data[data["age_months"] == age]

    speech_acts_children = data_age[data_age.speaker == CHILD]

    frequencies = calculate_frequencies(speech_acts_children[column_name_speech_act])

    # Filter for speech acts analyzed
    frequencies = {s: f for s, f in frequencies.items() if s in speech_acts_analyzed}

    # Add missing entries
    frequencies_complete = dict.fromkeys(speech_acts_analyzed, 0)
    frequencies_complete.update(frequencies)

    results = []
    for s, f in frequencies_complete.items():
        results.append(
            {
                "source": source,
                "speech_act": s,
                "frequency": f,
            }
        )

    results = pd.DataFrame(results)
    results.sort_values(by=["speech_act"], inplace=True)

    return results


def reproduce_speech_act_distribution(data, data_whole_childes):
    speech_acts_analyzed = [
        "YY",
        "ST",
        "PR",
        "MK",
        "SA",
        "RT",
        "RP",
        "RD",
        "AA",
        "AD",
        "AC",
        "QN",
        "YQ",
        "CL",
        "SI",
    ]

    fig, axes = plt.subplots(3, 1, sharex="all", sharey="all")

    for i, age in enumerate(AGES):
        results_snow = calculate_freq_distributions(
            data, SPEECH_ACT, speech_acts_analyzed, age, SOURCE_SNOW_LABEL
        )
        results_crf = calculate_freq_distributions(
            data, "y_pred", speech_acts_analyzed, age, SOURCE_AUTOMATIC_NEW_ENGLAND_LABEL
        )
        results_childes = calculate_freq_distributions(
            data_whole_childes,
            "y_pred",
            speech_acts_analyzed,
            age,
            SOURCE_AUTOMATIC_CHILDES_LABEL,
        )

        # Calculate KL divergences:
        kl_divergence = entropy(
            results_crf.frequency.to_list(), qk=results_snow.frequency.to_list()
        )
        print(f"KL Divergence (NewEngland, {age} months): {kl_divergence:.3f}")

        kl_divergence = entropy(
            results_childes.frequency.to_list(), qk=results_snow.frequency.to_list()
        )
        print(f"KL Divergence (CHILDES, {age} months): {kl_divergence:.3f}")

        jensen_shannon_distance = jensenshannon(
            results_crf.frequency.to_list(), results_snow.frequency.to_list()
        )
        print(
            f"Jensen-Shannon Distance (NewEngland, {age} months): {jensen_shannon_distance:.3f}"
        )

        jensen_shannon_distance = jensenshannon(
            results_childes.frequency.to_list(), results_snow.frequency.to_list()
        )
        print(
            f"Jensen-Shannon Distance (CHILDES, {age} months): {jensen_shannon_distance:.3f}"
        )

        results = pd.concat([results_snow, results_crf, results_childes])

        sns.barplot(
            ax=axes[i],
            x="speech_act",
            hue="source",
            y="frequency",
            data=results,
            hue_order=ORDER,
            clip_on=True,
        )

        # Move title into figure
        plt.rcParams["axes.titley"] = 1.0
        plt.rcParams["axes.titlepad"] = -14

        axes[i].set_title(f"Age: {age} months")

        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

        if i == 0:
            axes[i].legend(loc="upper left", bbox_to_anchor=(0, 0.79))
        else:
            axes[i].legend_.remove()

        axes[i].set_ylim(0, 0.3)

    axes[1].set_ylabel("frequency")
    axes[-1].set_xlabel("speech act")
    plt.tight_layout()
    plt.show()


def reproduce_speech_act_age_of_acquisition(data, data_whole_childes, target_measure=TARGET_PRODUCTION):
    if target_measure == TARGET_PRODUCTION:
        observed_speech_acts = AGE_OF_ACQUISITION_SPEECH_ACTS_ENOUGH_DATA
    else:
        observed_speech_acts = COMPREHENSION_SPEECH_ACTS_ENOUGH_DATA_2_OCCURRENCES

    ages_of_acquisition_snow = calc_ages_of_acquisition(
        target_measure,
        data,
        observed_speech_acts,
        AGES,
        data_source=SOURCE_SNOW,
        add_extra_datapoints=False,
        max_age=MAX_AGE,
        threshold_speech_act_observed_production=0,
    )

    ages_of_acquisition_crf = calc_ages_of_acquisition(
        target_measure,
        data,
        observed_speech_acts,
        AGES,
        data_source=SOURCE_CRF,
        add_extra_datapoints=False,
        max_age=MAX_AGE,
        threshold_speech_act_observed_production=0,
    )

    ages_of_acquisition_childes = calc_ages_of_acquisition(
        target_measure,
        data_whole_childes,
        observed_speech_acts,
        AGES,
        SOURCE_CRF,
        add_extra_datapoints=False,
        max_age=MAX_AGE,
        threshold_speech_act_observed_production=0,
    )

    # observed_speech_acts = list(data_whole_childes["y_pred"].dropna().unique())
    # ages_of_acquisition_childes_long = calc_ages_of_acquisition(
    #     target_measure,
    #     data_whole_childes,
    #     observed_speech_acts,
    #     AGES_LONG,
    #     SOURCE_CRF,
    #     add_extra_datapoints=False,
    #     max_age=MAX_AGE,
    #     threshold_speech_act_observed_production=0,
    # )

    aoa_data = []
    for speech_act in observed_speech_acts:
        aoa_data.append(
            {
                "speech_act": speech_act,
                "snow": ages_of_acquisition_snow[speech_act],
                "crf": ages_of_acquisition_crf[speech_act],
                "childes": ages_of_acquisition_childes[speech_act],
                # "childes (long)": ages_of_acquisition_childes_long[speech_act],
            }
        )
    aoa_data = pd.DataFrame(aoa_data).sort_values(by=["speech_act"], axis=0)
    print(aoa_data)
    print(aoa_data.to_latex(float_format="%.1f", index=False))

    path = f"results/aoa_correlations.csv"
    aoa_data.to_csv(path)

    # Calculate Spearman rank-order correlation
    corr_new_england = spearmanr(
        list(ages_of_acquisition_snow.values()), list(ages_of_acquisition_crf.values())
    )
    print("Spearman snow vs. crf: ", corr_new_england)
    corr_childes = spearmanr(
        list(ages_of_acquisition_snow.values()),
        list(ages_of_acquisition_childes.values()),
    )
    print("Spearman snow vs. childes: ", corr_childes)

    # Plot correlations
    fig, (axes) = plt.subplots(1, 2, sharex="all", sharey="all", figsize=(6, 3))

    axes[0].scatter(
        list(ages_of_acquisition_snow.values()),
        list(ages_of_acquisition_crf.values()),
        marker="",
    )
    axes[0].set_ylabel("CRF (New England)")
    axes[0].set_title(f"r={corr_new_england.correlation:.2f}")

    for speech_act in observed_speech_acts:
        axes[0].annotate(
            speech_act,
            (ages_of_acquisition_snow[speech_act], ages_of_acquisition_crf[speech_act]),
        )

    axes[1].scatter(
        list(ages_of_acquisition_snow.values()),
        list(ages_of_acquisition_childes.values()),
        marker="",
    )
    axes[1].set_ylabel("CRF (CHILDES)")
    axes[1].set_title(f"r={corr_childes.correlation:.2f}")

    for speech_act in observed_speech_acts:
        axes[1].annotate(
            speech_act,
            (
                ages_of_acquisition_snow[speech_act],
                ages_of_acquisition_childes[speech_act],
            ),
        )

    for i, ax in enumerate(axes):
        ax.set_xlabel("")

        axes[i].set_ylim(14, 60)
        if target_measure == TARGET_COMPREHENSION:
            axes[i].set_xlim(14, 40)
        else:
            axes[i].set_xlim(14, 60)

    fig.text(0.5, 0.04, "Data from Snow et al. (1996)", ha="center")
    plt.subplots_adjust(bottom=0.15)

    plt.show()


def convert_to_ranks(ages_of_acquisition):
    indices = sorted(ages_of_acquisition.values())
    ages_of_acquisition_rank = {}
    for speech_act, aoa in ages_of_acquisition.items():
        ages_of_acquisition_rank[speech_act] = indices.index(aoa)
    return ages_of_acquisition_rank


if __name__ == "__main__":
    print("Loading data...")
    data = pickle.load(open(PATH_NEW_ENGLAND_UTTERANCES_ANNOTATED, "rb"))

    # map ages to corresponding bins
    data["age_months"] = data["age_months"].apply(age_bin)

    # Load annotated data for whole CHILDES
    data_whole_childes = load_whole_childes_data()

    print(
        "Number of analyzed transcripts in CHILDES: ",
        len(
            data_whole_childes[
                (data_whole_childes.age_months >= min(AGES))
                & (data_whole_childes.age_months <= max(AGES))
            ].file_id.unique()
        ),
    )
    print(
        "Number of analyzed children in CHILDES: ",
        len(
            data_whole_childes[
                (data_whole_childes.age_months >= min(AGES))
                & (data_whole_childes.age_months <= max(AGES))
            ].child_id.unique()
        ),
    )

    reproduce_speech_act_age_of_acquisition(data, data_whole_childes, TARGET_PRODUCTION)

    reproduce_speech_act_age_of_acquisition(data, data_whole_childes, TARGET_COMPREHENSION)

    reproduce_speech_act_distribution(data, data_whole_childes)

    reproduce_num_speech_acts(data, data_whole_childes)
