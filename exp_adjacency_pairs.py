import pandas as pd
import matplotlib
from sklearn.preprocessing import OrdinalEncoder
import random

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

from utils import SOURCE_SNOW, SOURCE_CRF, SPEECH_ACT, CHILD, ADULT, SPEECH_ACT_DESCRIPTIONS, \
    PATH_NEW_ENGLAND_UTTERANCES_ANNOTATED, PATH_NEW_ENGLAND_UTTERANCES

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

ds_list = {
    SOURCE_SNOW: PATH_NEW_ENGLAND_UTTERANCES,
    SOURCE_CRF: PATH_NEW_ENGLAND_UTTERANCES_ANNOTATED,
}

# Colors
hex_colors_dic = {}
rgb_colors_dic = {}
hex_colors_only = []
for name, hex in matplotlib.colors.cnames.items():
    hex_colors_only.append(hex)
    hex_colors_dic[name] = hex
    rgb_colors_dic[name] = matplotlib.colors.to_rgb(hex)

ILL = SPEECH_ACT_DESCRIPTIONS.reset_index()
ILL["spa_2a"] = ILL["Category"].apply(lambda x: x[:3].upper())
node_colors_2a = {
    x: random.choice(hex_colors_only) for x in ILL["spa_2a"].unique().tolist()
}
ILL["colors"] = ILL["spa_2a"].apply(
    lambda x: None if x not in node_colors_2a.keys() else node_colors_2a[x]
)
node_colors_2 = (
    ILL[["Code", "colors"]].set_index("Code").to_dict()["colors"]
)  # no duplicates

ILL["concat"] = ILL.apply(lambda x: f"{x.Category} - {x.Description}", axis=1)
node_descr = ILL[["Code", "concat"]].set_index("Code").to_dict()["concat"]

###### SANKEY / PARSING FUNCTIONS
def plot_sankey(
    node_labels: list,
    node_colors: list,
    link_source: list,
    link_target: list,
    link_value: list,
    sk_title: str,
    node_customdata: list = None,
    link_customdata: list = None,
):
    d_node = dict(
        pad=15,
        thickness=15,
        line=dict(color="black", width=0.5),
        label=node_labels,
        color=node_colors,
    )
    d_link = dict(
        source=link_source,
        target=link_target,
        value=link_value,
    )

    if node_customdata is not None:
        d_node["customdata"] = node_customdata
        d_node["hovertemplate"] = "%{label}: %{customdata}<extra>%{value}</extra>"
        if link_customdata is not None:
            d_link["customdata"] = link_customdata
            d_link[
                "hovertemplate"
            ] = "Link from node %{source.customdata}<br /> to node%{target.customdata}<br />has value %{value} <br />and data %{customdata}<extra></extra>"

    fig = go.Figure(
        data=[go.Sankey(valueformat=".0f", valuesuffix="TWh", node=d_node, link=d_link)]
    )

    fig.update_layout(
        hovermode="x",
        title=sk_title,
        font=dict(size=25, color="black"),
    )

    return fig

SPEAKER_SOURCE = "source"
SPEAKER_TARGET = "target"


def get_adj_pairs_frac_data(data, age, source: str = ADULT,
    target: str = CHILD,
    min_percent: float = 0.0,
    min_percent_recipient: float = 0.0,
    data_source = SOURCE_SNOW):

    if data_source == SOURCE_SNOW:
        column_name_speech_act = SPEECH_ACT
    elif data_source == SOURCE_CRF:
        column_name_speech_act = "y_pred"
    else:
        raise ValueError("Unknown data source: ", data_source)

    spa_seq = gen_seq_data(data, age=age, column_name_speech_act=column_name_speech_act)

    # for now source = 1 and target = 0
    spa_seq.rename(
        {"speaker_code_1": "source", "speaker_code_0": "target"}, axis="columns", inplace=True
    )

    spa_source = column_name_speech_act +"_1"
    spa_target = column_name_speech_act +"_0"

    # 1. Choose illocutionary or interchange, remove unused sequences, remove NAs, select direction (MOT => CHI or CHI => MOT)
    spa_seq.dropna(how="any", inplace=True)
    if source is not None and source in [CHILD, ADULT]:
        spa_seq = spa_seq[(spa_seq[SPEAKER_TARGET] == target)]
    if target is not None and target in [CHILD, ADULT]:
        spa_seq = spa_seq[(spa_seq[SPEAKER_SOURCE] == source)]
    # 2. Groupby, unstack and orderby
    spa_gp = (
        spa_seq.groupby(by=[spa_target, spa_source])
            .agg({SPEAKER_TARGET: "count"})
            .reset_index(drop=False)
    )
    # 3. Filter out infrequent sequences
    spa_gp["v_percent"] = spa_gp[SPEAKER_TARGET] / spa_gp[SPEAKER_TARGET].sum()
    spa_gp = spa_gp[spa_gp["v_percent"] >= min_percent].reset_index(drop=True)

    percentages = []
    # Save frequency data
    for speech_act_source in spa_gp[spa_source].unique():
        speech_acts_target = spa_gp[spa_gp[spa_source] == speech_act_source]
        for speech_act_target in speech_acts_target[spa_target]:
            count = speech_acts_target[
                speech_acts_target[spa_target] == speech_act_target
                ][SPEAKER_TARGET].values[0]
            fraction = count / speech_acts_target[SPEAKER_TARGET].sum()
            percentages.append(
                {
                    SPEAKER_SOURCE: speech_act_source,
                    SPEAKER_TARGET: speech_act_target,
                    "fraction": fraction,
                }
            )
    percentages = pd.DataFrame(percentages)
    percentages["source_description"] = percentages["source"].apply(
        lambda sp: SPEECH_ACT_DESCRIPTIONS.loc[sp].Description
    )
    percentages["target_description"] = percentages["target"].apply(
        lambda sp: SPEECH_ACT_DESCRIPTIONS.loc[sp].Description
    )

    percentages = percentages[percentages["fraction"] > min_percent_recipient]

    return percentages, spa_gp


def gen_seq_data(data, age: int = None, column_name_speech_act = "speech_act"):
    # 0. Choose age
    if age is not None:
        data_age = data[data.age == age]
    # 1. Sequence extraction & columns names
    spa_shifted = {0: data_age[[column_name_speech_act, "speaker_code", "transcript_file"]]}
    spa_shifted[1] = (
        spa_shifted[0]
        .shift(periods=1, fill_value=None)
        .rename(columns={col: col + "_1" for col in spa_shifted[0].columns})
    )
    spa_shifted[0] = spa_shifted[0].rename(
        columns={col: col + "_0" for col in spa_shifted[0].columns}
    )
    # 2. Merge
    spa_compare = pd.concat(spa_shifted.values(), axis=1)
    # 3. Add empty slots for file changes
    spa_compare.loc[
        (spa_compare["transcript_file_0"] != spa_compare["transcript_file_1"]), [f"{column_name_speech_act}_1"]
    ] = None
    return spa_compare[[col for col in spa_compare.columns if "transcript_file" not in col]]


def create_sankey_diagram(
    spa_gp,
    age,
    source: str = ADULT,
    target: str = CHILD,
    column_name_speech_act = "speech_act"
):
    spa_source = column_name_speech_act + "_1"
    spa_target = column_name_speech_act + "_0"

    # 5.1 Apply encoder to get labels as numbers => idx in sankey (source, target)
    enc = OrdinalEncoder()
    trf_spa = pd.DataFrame(
        enc.fit_transform(spa_gp[[spa_target, spa_source]]),
        columns=["target", "source"],
    )
    enc_cat = {
        col: list(ar) for col, ar in zip([spa_target, spa_source], enc.categories_)
    }

    trf_spa[["value", "v_percent"]] = spa_gp[[SPEAKER_TARGET, "v_percent"]]
    # 5.2 Add link colors
    # 5.3 Update categories for target columns
    n = len(enc_cat[spa_target])
    trf_spa["source"] = trf_spa["source"] + n
    # 6.2 Plot
    fig = plot_sankey(
        node_labels=(enc_cat[spa_target] + enc_cat[spa_source]),
        node_colors=[
            node_colors_2[x] for x in (enc_cat[spa_target] + enc_cat[spa_source])
        ],
        link_source=trf_spa["source"],
        link_target=trf_spa["target"],
        link_value=trf_spa["value"],
        node_customdata=[
            node_descr[x] for x in (enc_cat[spa_target] + enc_cat[spa_source])
        ],
        sk_title=f"{source} to {target} adjacency pairs | Child age: {age} months",
    )

    return fig


###### LAYOUT
app.layout = html.Div(
    children=[
        html.H1(children="CHILDES - Analysis of Parent-Children Speech Acts"),
        html.Div(
            [
                html.Div(
                    children=[
                        "Pick a data source:",
                        dcc.Dropdown(
                            id="dataset-choice",
                            options=[{"label": i, "value": i} for i in ds_list.keys()],
                            value=SOURCE_SNOW,
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                html.Div(
                    children=[
                        "source",
                        dcc.Dropdown(
                            id="source",
                            options=[{"label": i, "value": i} for i in [CHILD, ADULT]],
                            value=ADULT,
                        ),
                        "target",
                        dcc.Dropdown(
                            id="target",
                            options=[{"label": i, "value": i} for i in [CHILD, ADULT]],
                            value=CHILD,
                        ),
                        "child age",
                        dcc.Dropdown(
                            id="age",
                            options=[{"label": i, "value": i} for i in [14, 20, 32]],
                            value=32,
                        ),
                        "percentage",
                        dcc.Dropdown(
                            id="percentage",
                            options=[
                                {"label": i, "value": i}
                                for i in [
                                    0,
                                    0.001,
                                    0.005,
                                    0.01,
                                    0.012,
                                    0.015,
                                    0.02,
                                    0.025,
                                ]
                            ],
                            value=0.01,
                        ),
                    ],
                    style={"width": "24%", "display": "inline-block"},
                ),
            ]
        ),
        dcc.Graph(id="sankey"),  # will be updated through callbacks
    ]
)

###### CALLBACKS
@app.callback(
    Output("sankey", "figure"),
    [
        Input("dataset-choice", "value"),
        Input("source", "value"),
        Input("target", "value"),
        Input("age", "value"),
        Input("percentage", "value"),
    ],
)
def update_graph(dataset, source, target, age, percentage):
    # Load data
    data = pd.read_pickle(ds_list[dataset])
    match_age = [14, 20, 32]
    data["age"] = data.age.apply(
        lambda age: min(match_age, key=lambda x: abs(x - age))
    )
    # Filter data
    _, spa_gp = get_adj_pairs_frac_data(data, age, source, target, min_percent=percentage, data_source=dataset)

    if dataset == SOURCE_SNOW:
        column_name_speech_act = SPEECH_ACT
    elif dataset == SOURCE_CRF:
        column_name_speech_act = "y_pred"
    else:
        raise ValueError("Unknown data source: ", dataset)

    fig = create_sankey_diagram(spa_gp, age, source, target, column_name_speech_act)

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
