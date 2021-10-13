import pickle

import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, pearsonr

from age_of_acquisition import MAX_AGE, calc_ages_of_acquisition, COMPREHENSION_SPEECH_ACTS_ENOUGH_DATA_2_OCCURRENCES
from exp_reproduce_snow import AGE_OF_ACQUISITION_SPEECH_ACTS_ENOUGH_DATA
from utils import TARGET_PRODUCTION, age_bin, AGES, SOURCE_SNOW, \
    TARGET_COMPREHENSION, PATH_NEW_ENGLAND_UTTERANCES

if __name__ == "__main__":
    data = pickle.load(open(PATH_NEW_ENGLAND_UTTERANCES, "rb"))

    # map ages to corresponding bins
    data["age_months"] = data["age_months"].apply(age_bin)

    observed_speech_acts = AGE_OF_ACQUISITION_SPEECH_ACTS_ENOUGH_DATA

    aoa_production = calc_ages_of_acquisition(
        TARGET_PRODUCTION,
        data,
        observed_speech_acts,
        AGES,
        data_source=SOURCE_SNOW,
        add_extra_datapoints=False,
        max_age=MAX_AGE,
        threshold_speech_act_observed_production=0,
    )

    observed_speech_acts = COMPREHENSION_SPEECH_ACTS_ENOUGH_DATA_2_OCCURRENCES

    aoa_comprehension = calc_ages_of_acquisition(
        TARGET_COMPREHENSION,
        data,
        observed_speech_acts,
        AGES,
        data_source=SOURCE_SNOW,
        add_extra_datapoints=False,
        max_age=MAX_AGE,
        threshold_speech_act_observed_production=0,
    )

    aoa_production = pd.DataFrame(aoa_production, index=[0]).T.reset_index().rename(columns={"index": "speech_act", 0: "age_of_acquisition"})
    aoa_comprehension = pd.DataFrame(aoa_comprehension, index=[0]).T.reset_index().rename(columns={"index": "speech_act", 0: "age_of_acquisition"})

    # Drop values that are max_val
    aoa_production = aoa_production[aoa_production.age_of_acquisition < MAX_AGE]
    aoa_comprehension = aoa_comprehension[aoa_comprehension.age_of_acquisition < MAX_AGE]

    aoa_production["measure"] = "Production"
    aoa_comprehension["measure"] = "Comprehension"

    aoa_all = aoa_comprehension.append(aoa_production, ignore_index=True)

    aoa_comprehension = aoa_comprehension[aoa_comprehension.speech_act.isin(aoa_production.speech_act)].sort_values(by="speech_act")
    aoa_production = aoa_production[aoa_production.speech_act.isin(aoa_comprehension.speech_act)].sort_values(by="speech_act")

    print("Paired t-test: ", ttest_rel(aoa_comprehension.age_of_acquisition.values, aoa_production.age_of_acquisition.values))
    # print(ttest_ind(aoa_comprehension.age_of_acquisition.values, aoa_production.age_of_acquisition.values))

    print("Pearson's r: ", pearsonr(aoa_comprehension.age_of_acquisition.values, aoa_production.age_of_acquisition.values))

    plt.figure()

    sns.barplot(data=aoa_all, x="measure", y="age_of_acquisition", ci=None, alpha=0.7)
    sns.swarmplot(data=aoa_all, x="measure", y="age_of_acquisition", color="black")

    plt.ylabel("Age of acquisition (months)")

    plt.savefig("../figures/aoa_comprehension_vs_production.png", dpi=300)
    plt.show()
