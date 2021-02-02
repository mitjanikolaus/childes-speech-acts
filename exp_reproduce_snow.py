import math
import pickle

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
from scipy.stats import spearmanr

from age_of_acquisition import TARGET_PRODUCTION, calc_ages_of_acquisition
from preprocess import SPEECH_ACT, CHILD
from utils import age_bin, calculate_frequencies

AGES = [14, 20, 32]
AGES_LONG = [14, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54]

SOURCE_SNOW = "Data from Snow et al. (1996)"
SOURCE_AUTOMATIC_NEW_ENGLAND = "Automatically Annotated Data (New England)"
SOURCE_AUTOMATIC_CHILDES = "Automatically Annotated Data (English CHILDES)"
ORDER = [SOURCE_SNOW, SOURCE_AUTOMATIC_NEW_ENGLAND, SOURCE_AUTOMATIC_CHILDES]

MAX_NUM_SPEECH_ACT_TYPES = 25

TRANSCRIPTS_NEW_ENGLAND = [3580, 3581, 3582, 3583, 3584, 3585, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3639, 3640, 3641, 3642, 3643, 3644, 3645, 3646, 3647, 3648, 3649, 3650, 3651, 3652, 3653, 3654, 3655, 3656, 3657, 3658, 3659, 3660, 3661, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3669, 3670, 3671, 3672, 3673, 3674, 3675, 3676, 3677, 3678, 3679, 3680, 3681, 3682, 3683, 3684, 3685, 3686, 3687, 3688, 3689, 3690, 3691, 3692, 3693, 3694, 3695, 3696, 3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710, 3711, 3712, 3713, 3714, 3715, 3716, 3717, 3718, 3719, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3727, 3728, 3729, 3730, 3731, 3732, 3733, 3734, 3735, 3736, 3737, 3738, 3739, 3740, 3741, 3742, 3743, 3744, 3745, 3746, 3747, 3748, 3749, 3750, 3751, 3752, 3753, 3754, 3755, 3756, 3757]

AGE_OF_ACQUISITION_MIN_DATAPOINTS = 2
AGE_OF_ACQUISITION_SPEECH_ACTS_ENOUGH_DATA = ['RP', 'QN', 'TO', 'ST', 'RR', 'SA', 'AP', 'AC', 'YQ', 'MK', 'CL', 'PF', 'SI', 'RT', 'PR', 'RD', 'FP', 'AD', 'CS', 'DC', 'AN', 'PA', 'DW', 'AA', 'SC']
AGE_OF_ACQUISITION_SPEECH_ACTS_ENOUGH_DATA_NO_DECLINE = ['RP', 'QN', 'ST', 'RR', 'SA', 'AP', 'AC', 'YQ', 'MK', 'CL', 'SI', 'RT', 'PR', 'RD', 'FP', 'AD', 'CS', 'DC', 'AN', 'PA', 'DW', 'AA', 'SC']
AGE_OF_ACQUISITION_MAX_AGE = 12 * 18

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

        # Normalize children counts to get fractions
        for num_speech_act_types in results_age.keys():
            results_age[num_speech_act_types] = results_age[num_speech_act_types] / len(children_ids)

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
    results_snow["source"] = SOURCE_SNOW

    results_crf = calculate_num_speech_act_types(data, "y_pred")
    results_crf["source"] = SOURCE_AUTOMATIC_NEW_ENGLAND

    results_childes = calculate_num_speech_act_types(
        data_whole_childes, "y_pred"
    )
    results_childes["source"] = SOURCE_AUTOMATIC_CHILDES

    results = results_snow.append(results_crf).append(results_childes)

    fig, (axes) = plt.subplots(3, 1, sharex="all")

    # Move title into figure
    plt.rcParams["axes.titley"] = 1.0
    plt.rcParams["axes.titlepad"] = -14

    for i, age in enumerate(AGES):
        results_age = results[results.age == age]

        sns.barplot(ax=axes[i], x="num_types", hue="source", y="frac_children", data=results_age, hue_order=ORDER)

        if i == 0:
            axes[i].legend(loc="upper right", bbox_to_anchor=(1, 0.8))
        else:
            axes[i].legend_.remove()

        axes[i].set_title(f"Age: {age} months")

        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

        if age > 14:
            axes[i].set_ylim(0, 0.23)

    axes[1].set_ylabel("fraction of children")
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

    results = []
    for s, f in frequencies.items():
        results.append(
            {
                "source": source,
                "speech_act": s,
                "frequency": f,
            }
        )

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

    fig, axes = plt.subplots(3, 1, sharex="all")

    for i, age in enumerate(AGES):
        results_snow = calculate_freq_distributions(
            data, SPEECH_ACT, speech_acts_analyzed, age, SOURCE_SNOW
        )
        results_crf = calculate_freq_distributions(
            data, "y_pred", speech_acts_analyzed, age, SOURCE_AUTOMATIC_NEW_ENGLAND
        )
        results_childes = calculate_freq_distributions(
            data_whole_childes, "y_pred", speech_acts_analyzed, age, SOURCE_AUTOMATIC_CHILDES
        )

        results = pd.DataFrame(results_snow + results_crf + results_childes)
        results.sort_values(by=["speech_act"], inplace=True)

        sns.barplot(
            ax=axes[i], x="speech_act", hue="source", y="frequency", data=results, hue_order=ORDER
        )

        # Move title into figure
        plt.rcParams["axes.titley"] = 1.0
        plt.rcParams["axes.titlepad"] = -14

        axes[i].set_title(f"Age: {age} months")

        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

        if i == 0:
            axes[i].legend(loc="upper left", bbox_to_anchor=(0, 0.8))
        else:
            axes[i].legend_.remove()

        if age > 14:
            axes[i].set_ylim(0, 0.3)

    axes[1].set_ylabel("frequency")
    axes[-1].set_xlabel("speech act")
    plt.tight_layout()
    plt.show()


def reproduce_speech_act_age_of_acquisition(data, data_whole_childes):
    observed_speech_acts = AGE_OF_ACQUISITION_SPEECH_ACTS_ENOUGH_DATA

    ages_of_acquisition_snow = calc_ages_of_acquisition(
        TARGET_PRODUCTION,
        data,
        observed_speech_acts,
        AGES,
        SPEECH_ACT,
        add_extra_datapoints=False,
        max_age=AGE_OF_ACQUISITION_MAX_AGE,
        threshold_speech_act_observed_production=0,
    )

    ages_of_acquisition_crf = calc_ages_of_acquisition(
        TARGET_PRODUCTION,
        data,
        observed_speech_acts,
        AGES,
        "y_pred",
        add_extra_datapoints=False,
        max_age=AGE_OF_ACQUISITION_MAX_AGE,
        threshold_speech_act_observed_production=0,
    )

    ages_of_acquisition_childes = calc_ages_of_acquisition(
        TARGET_PRODUCTION,
        data_whole_childes,
        observed_speech_acts,
        AGES,
        "y_pred",
        add_extra_datapoints=False,
        max_age=AGE_OF_ACQUISITION_MAX_AGE,
        threshold_speech_act_observed_production=0,
    )

    # ages_of_acquisition_childes_long = calc_ages_of_acquisition(
    #     TARGET_PRODUCTION,
    #     data_whole_childes,
    #     observed_speech_acts,
    #     AGES_LONG,
    #     "y_pred",
    #     add_extra_datapoints=False,
    #     max_age=AGE_OF_ACQUISITION_MAX_AGE,
    #     threshold_speech_act_observed_production=0,
    # )

    aoa_data = []
    for speech_act in observed_speech_acts:
        aoa_data.append({
            "speech_act": speech_act,
            "snow": ages_of_acquisition_snow[speech_act],
            "crf": ages_of_acquisition_crf[speech_act],
            "childes": ages_of_acquisition_childes[speech_act],
            # "childes (long)": ages_of_acquisition_childes_long[speech_act],
        })
    aoa_data = pd.DataFrame(aoa_data)
    print(aoa_data)
    print(aoa_data.to_latex(float_format="%.1f", index=False))

    path = f"results/aoa_correlations.csv"
    aoa_data.to_csv(path)

    # Calculate Spearman rank-order correlation
    corr_new_england = spearmanr(list(ages_of_acquisition_snow.values()), list(ages_of_acquisition_crf.values()))
    print("Spearman snow vs. crf: ", corr_new_england)
    corr_childes = spearmanr(list(ages_of_acquisition_snow.values()), list(ages_of_acquisition_childes.values()))
    print("Spearman snow vs. childes: ", corr_childes)
    # corr_childes_dense = spearmanr(list(ages_of_acquisition_snow.values()), list(ages_of_acquisition_childes_long.values()))
    # print("Spearman snow vs. childes (long): ", corr_childes_dense)

    # Plot correlations
    fig, (axes) = plt.subplots(1, 2, sharex="all", sharey="all", figsize=(6, 3))

    axes[0].scatter(list(ages_of_acquisition_snow.values()), list(ages_of_acquisition_crf.values()))
    axes[0].set_ylabel("CRF (New England)")
    axes[0].set_title(f"r={corr_new_england.correlation:.2f}")

    axes[1].scatter(list(ages_of_acquisition_snow.values()), list(ages_of_acquisition_childes.values()))
    axes[1].set_ylabel("CRF (CHILDES)")
    axes[1].set_title(f"r={corr_childes.correlation:.2f}")

    # axes[2].scatter(list(ages_of_acquisition_snow.values()), list(ages_of_acquisition_childes_dense.values()))
    # axes[2].set_ylabel("CRF (CHILDES, dense)")
    # axes[2].set_title(f"r={corr_childes_dense.correlation:.2f}")

    for i, ax in enumerate(axes):
        ax.set_xlabel("")

        axes[i].set_ylim(14, 60)
        axes[i].set_xlim(14, 60)

    fig.text(0.5, 0.04, "Data from Snow et al. (1996)", ha='center')
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
    data = pickle.load(open("data/new_england_reproduced_crf.p", "rb"))

    # map ages to corresponding bins
    data["age_months"] = data["age_months"].apply(age_bin)

    # calculate minimum number of utterances for each age group
    min_num_utterances = {}
    for age in AGES:
        data_age = data[(data.age_months == age) & (data.speaker == CHILD)]
        lengths = data_age.groupby(by=["file_id"]).agg(length=("utterance_id", lambda x: len(x)))
        min_num_utterances[age] = lengths.length.min()
    print("Min num utterances: ", min_num_utterances)

    # Load annotated data for whole CHILDES
    data_whole_childes = pd.read_hdf("~/data/speech_acts/data/speech_acts_chi.h5")

    # Filter out New England corpus transcripts
    data_whole_childes = data_whole_childes[~data_whole_childes.file_id.isin(TRANSCRIPTS_NEW_ENGLAND)]

    # Filter for min num utterances
    for age in AGES:
        lengths = data_whole_childes[data_whole_childes.age_months == age].groupby(by=["file_id"]).agg(length=("file_id", lambda x: len(x)))
        transcripts_too_short = lengths[lengths.length < min_num_utterances[age]].index.to_list()

        print(f"Filtering out {len(transcripts_too_short)} transcripts that are too short (age {age} months)")
        data_whole_childes = data_whole_childes[~data_whole_childes.file_id.isin(transcripts_too_short)]

    print("Number of analyzed transcripts in CHILDES: ", len(data_whole_childes[(data_whole_childes.age_months >= min(AGES)) & (data_whole_childes.age_months <= max(AGES))].file_id.unique()))
    print("Number of analyzed children in CHILDES: ", len(data_whole_childes[(data_whole_childes.age_months >= min(AGES)) & (data_whole_childes.age_months <= max(AGES))].child_id.unique()))

    # reproduce_speech_act_age_of_acquisition(data, data_whole_childes)

    # reproduce_speech_act_distribution(data, data_whole_childes)

    reproduce_num_speech_acts(data, data_whole_childes)
