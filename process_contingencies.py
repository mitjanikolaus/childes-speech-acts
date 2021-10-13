import pandas as pd

from utils import SOURCE_SNOW, SOURCE_CRF, load_whole_childes_data, age_bin, PATH_NEW_ENGLAND_UTTERANCES_ANNOTATED
from exp_adjacency_pairs import get_adj_pairs_frac_data
from utils import AGES, ADULT, CHILD


def get_contingency_data(data, age, data_source):
    contingency_data = pd.read_csv(
        "adjacency_pairs/adjacency_pairs_contingency.csv", keep_default_na=False
    )

    adj_data, _ = get_adj_pairs_frac_data(
        data,
        age,
        source=ADULT,
        target=CHILD,
        min_percent=0.0,
        min_percent_recipient=0.0,
        data_source=data_source,
    )

    contingencies = []
    for i, row in adj_data.iterrows():
        source, target = row["source"], row["target"]
        if (
                len(
                    contingency_data[
                        (contingency_data.source == source)
                        & (contingency_data.target == target)
                    ]
                )
                > 0
        ):
            cont = contingency_data[
                (contingency_data.source == source)
                & (contingency_data.target == target)
                ].contingency.values[0]
        else:
            print(f"Warning: Unknown speech act combination: {source}-{target}")
            cont = "TODO"
        contingencies.append(cont)

    adj_data["contingency"] = contingencies

    return adj_data


def create_file_with_all_possible_adjacency_pairs_for_annotation():
    adj_data_all = pd.DataFrame()

    data = pd.read_pickle(PATH_NEW_ENGLAND_UTTERANCES_ANNOTATED)

    # map ages to corresponding bins
    data["age_months"] = data["age_months"].apply(age_bin)

    for data_source in [SOURCE_SNOW, SOURCE_CRF]:
        for age in AGES:
            adj_data, _ = get_adj_pairs_frac_data(
                data,
                age,
                source=ADULT,
                target=CHILD,
                min_percent=0.0,
                min_percent_recipient=0.0,
                data_source=data_source,
            )
            adj_data_all = adj_data_all.append(adj_data, ignore_index=True)

    # whole childes data:
    data_whole_childes = load_whole_childes_data()

    for age in AGES:
        adj_data, _ = get_adj_pairs_frac_data(
            data_whole_childes,
            age,
            source=ADULT,
            target=CHILD,
            min_percent=0.0,
            min_percent_recipient=0.0,
            data_source=SOURCE_CRF,
        )
        adj_data_all = adj_data_all.append(adj_data, ignore_index=True)

    adj_data_all.drop_duplicates(subset=["source", "target"], inplace=True)
    to_annotate = adj_data_all[
        ["source", "target", "source_description", "target_description"]
    ]

    to_annotate.sort_values(by="source", inplace=True)

    to_annotate["contingency"] = to_annotate.apply(
        lambda row: 0 if row["target"] in ["YY", "OO"] else "TODO", axis=1
    )

    to_annotate.to_csv("data/adjacency_pairs_for_annotation.csv", index=False)


if __name__ == "__main__":
    create_file_with_all_possible_adjacency_pairs_for_annotation()
