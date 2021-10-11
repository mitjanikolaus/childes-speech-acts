import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind

from age_of_acquisition import MAX_AGE

if __name__ == "__main__":
    aoa_production = pd.read_csv("results/age_of_acquisition_production.csv")
    aoa_comprehension = pd.read_csv("results/age_of_acquisition_comprehension.csv")

    # Drop values that are max_val
    aoa_production = aoa_production[aoa_production.age_of_acquisition < MAX_AGE]
    aoa_comprehension = aoa_comprehension[aoa_comprehension.age_of_acquisition < MAX_AGE]

    aoa_production["measure"] = "Production"
    aoa_comprehension["measure"] = "Comprehension"

    aoa_all = aoa_comprehension.append(aoa_production, ignore_index=True)

    # aoa_comprehension = aoa_comprehension[aoa_comprehension.speech_act.isin(aoa_production.speech_act)].sort_values(by="speech_act")
    # aoa_production = aoa_production[aoa_production.speech_act.isin(aoa_comprehension.speech_act)].sort_values(by="speech_act")

    # print(ttest_rel(aoa_comprehension.age_of_acquisition.values, aoa_production.age_of_acquisition.values))
    print(ttest_ind(aoa_comprehension.age_of_acquisition.values, aoa_production.age_of_acquisition.values))

    sns.barplot(data=aoa_all, x="measure", y="age_of_acquisition", ci=None, alpha=0.7)
    sns.swarmplot(data=aoa_all, x="measure", y="age_of_acquisition", color="black")

    plt.ylabel("Age of acquisition (months)")

    plt.savefig("results/aoa_comprehension_vs_production.png", dpi=300)
    plt.show()
