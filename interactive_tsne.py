import os
import threading
import http.server
import functools
import socketserver
import argparse

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, State, Input, Output
# from dash.dependencies import Input, Output
from name_lib import *

def serve_folder(folder, port=8888):
    """
    Serve all files in 'folder' on the given 'port'
    WITHOUT changing the global current working directory.
    """
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=folder)
    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"Serving files from '{folder}' at http://127.0.0.1:{port}")
        httpd.serve_forever()

def build_figure_for_l2(df_all, l2_word_label, image_port, embedding="tsne"):
    """
    Given the full DataFrame (df_all), filter rows for the specified L2 label,
    then build and return a Plotly scatter figure with proper x,y axes and labels
    based on the embedding type (t-SNE or UMAP).
    """
    # Filter based on L2 label.
    if l2_word_label is None or l2_word_label == "ALL":
        df_filtered = df_all.copy()
    else:
        selected_l2_label = NAME_LABEL_L2.get(l2_word_label)
        if selected_l2_label is None:
            return px.scatter()  # Return empty figure if unrecognized

        l3_ids = [
            l3_label for l3_word, (l3_label, parent_l2_label) in REASSIGN_NAME_LABEL_L3L2.items()
            if parent_l2_label == selected_l2_label
        ]
        df_filtered = df_all[df_all['label'].isin(l3_ids)]

    # Select coordinate columns based on embedding type.
    if embedding == "umap":
        xcol = "umap_x"
        ycol = "umap_y"
    else:
        xcol = "tsne_x"
        ycol = "tsne_y"

    # Define hover info and custom data.
    hover_columns = ["filename", "word_label"]
    # custom_data: we include image URL, filename, and ground-truth label.
    custom_data_cols = ["img_url", "filename", "word_label"]
    if "predicted_word_label" in df_filtered.columns:
        custom_data_cols.extend([
            "predicted_word_label", "top3_label_1", "top3_prob_1",
            "top3_label_2", "top3_prob_2", "top3_label_3", "top3_prob_3"
        ])

    fig = px.scatter(
        df_filtered,
        x=xcol,
        y=ycol,
        color="word_label",
        symbol="word_label",
        hover_data=hover_columns,
        custom_data=custom_data_cols
    )

    # Use the embedding type in the title and axis labels.
    title_prefix = embedding.upper()  # "UMAP" or "TSNE"
    if l2_word_label and l2_word_label != "ALL":
        title = f"{title_prefix} (L2: {l2_word_label}) - {len(df_filtered)} points"
    else:
        title = f"{title_prefix} (All data)"
    fig.update_layout(
        title=title,
        legend_title="L3 Word Label"
    )
    fig.update_xaxes(title=f"{title_prefix} X")
    fig.update_yaxes(title=f"{title_prefix} Y")
    return fig

def main():
    # 1) PARSE COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser(
        description="t-SNE/UMAP Viewer with local image hosting using Dash."
    )
    parser.add_argument(
        "--emb-results",  # You can also pass a UMAP file here.
        type=str,
        default="tsne_results.csv",
        help="Path to the CSV file containing embedding results. "
             "For t-SNE, expected columns: tsne_x, tsne_y, label, filename. "
             "For UMAP, expected columns: umap_x, umap_y, label, filename."
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        required=True,
        help="Path to the local folder containing images."
    )
    parser.add_argument(
        "--image-port",
        type=int,
        default=8888,
        help="Port number on which to serve images."
    )
    parser.add_argument(
        "--dash-port",
        type=int,
        default=8050,
        help="Port number on which to run the Dash app."
    )
    parser.add_argument(
        "--show-cls",
        default='Grassland',
        help="Name of L2 Class (e.g., Grassland)."
    )
    parser.add_argument(
        "--predictions-csv1",
        type=str,
        default=None,
        help="Path to first predictions CSV file."
    )
    parser.add_argument(
        "--predictions-csv2",
        type=str,
        default=None,
        help="Path to second predictions CSV file."
    )

    args = parser.parse_args()

    # 2) LOAD EMBEDDING RESULTS CSV
    emb_csv = os.path.abspath(args.emb_results)
    if not os.path.exists(emb_csv):
        raise FileNotFoundError(f"[ERROR] Results file not found: {emb_csv}")

    df = pd.read_csv(emb_csv)
    required_cols = {"label", "filename"}
    if "umap_x" in df.columns and "umap_y" in df.columns:
        embedding = "umap"
        required_cols.update({"umap_x", "umap_y"})
    else:
        # Default to t-SNE
        embedding = "tsne"
        required_cols.update({"tsne_x", "tsne_y"})

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # 3) Create a 'word_label' column from numeric L3 'label'
    df["word_label"] = df["label"].map(REASSIGN_LABEL_NAME_L3)

    # 4) Create an 'img_url' column for the samples using the image server URL.
    base_url = f"http://localhost:{args.image_port}/"
    df["img_url"] = base_url + df["filename"].astype(str)

    # 5) Merge predictions CSVs if provided (same as before)
    pred_dfs = []
    if args.predictions_csv1:
        pred_csv1 = os.path.abspath(args.predictions_csv1)
        if not os.path.exists(pred_csv1):
            raise FileNotFoundError(f"[ERROR] Predictions CSV1 file not found: {pred_csv1}")
        df_pred1 = pd.read_csv(pred_csv1)
        for col in ["top3_label_1", "top3_label_2", "top3_label_3"]:
            if col in df_pred1.columns:
                df_pred1[col] = df_pred1[col].map(REASSIGN_LABEL_NAME_L3)
        pred_dfs.append(df_pred1)
    if args.predictions_csv2:
        pred_csv2 = os.path.abspath(args.predictions_csv2)
        if not os.path.exists(pred_csv2):
            raise FileNotFoundError(f"[ERROR] Predictions CSV2 file not found: {pred_csv2}")
        df_pred2 = pd.read_csv(pred_csv2)
        for col in ["top3_label_1", "top3_label_2", "top3_label_3"]:
            if col in df_pred2.columns:
                df_pred2[col] = df_pred2[col].map(REASSIGN_LABEL_NAME_L3)
        pred_dfs.append(df_pred2)
    if pred_dfs:
        df_pred_all = pd.concat(pred_dfs, ignore_index=True)
        df = df.merge(df_pred_all, left_on="filename", right_on="file_name", how="left")
        if "file_name" in df.columns:
            df = df.drop(columns=["file_name"])

    # 6) Create the Dash app layout.
    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Store(id="full-data", data=df.to_dict(orient="records")),
        html.H3("Select L2 Class:"),
        dcc.Dropdown(
            id="l2-dropdown",
            options=[{"label": "ALL", "value": "ALL"}] + [
                {"label": l2_name, "value": l2_name}
                for l2_name in NAME_LABEL_L2.keys()
            ],
            value="ALL",
            clearable=False,
            style={"width": "300px"}
        ),
        html.Div([
            # Left: embedding figure.
            html.Div(
                dcc.Graph(
                    id="tsne-graph",  # The graph id stays the same.
                    style={"width": "100%", "height": "800px"}
                ),
                style={"width": "70%"}
            ),
            # Right: Combined information and image panels.
            html.Div([
                html.Div(
                    id="hover-info",
                    children="Click a point to see details.",
                    style={"marginBottom": "20px", "border": "1px solid #ccc", "padding": "10px"}
                ),
                html.Div([
                    html.Div([
                        html.H4("Raw Image"),
                        html.Img(
                            id="clicked-image",
                            style={
                                "width": "300px",
                                "height": "auto",
                                "border": "1px solid #ccc",
                                "padding": "5px"
                            }
                        )
                    ], style={"marginBottom": "20px"}),
                    html.Div([
                        html.H4("GradCAM Image"),
                        html.Img(
                            id="gradcam-image",
                            style={
                                "width": "300px",
                                "height": "auto",
                                "border": "1px solid #ccc",
                                "padding": "5px"
                            }
                        )
                    ])
                ])
            ], style={"width": "30%", "paddingLeft": "20px"})
        ], style={"display": "flex", "flexDirection": "row", "alignItems": "flex-start"})
    ])

    # 7) Callback to update the embedding figure.
    @app.callback(
        Output("tsne-graph", "figure"),
        [Input("l2-dropdown", "value")],
        [State("full-data", "data")]
    )
    def update_figure(l2_value, stored_data):
        df_full = pd.DataFrame(stored_data)
        fig = build_figure_for_l2(df_full, l2_value, args.image_port, embedding)
        return fig

    # 8) Combined callback to update images and info panel on click.
    @app.callback(
        [Output("clicked-image", "src"),
         Output("gradcam-image", "src"),
         Output("hover-info", "children")],
        [Input("tsne-graph", "clickData")]
    )
    def update_click_info(clickData):
        if clickData and "points" in clickData:
            point = clickData["points"][0]
            cd = point["customdata"]
            # Custom data indexing:
            # cd[0] = raw image URL, cd[1] = filename, cd[2] = ground-truth (word_label),
            # and if available, cd[3] ... cd[9] = prediction info.
            raw_img_url = cd[0]
            filename = cd[1]
            ground_truth = cd[2]
            base = raw_img_url.rsplit('/', 1)[0]
            gradcam_img_url = f"{base}/cam_{filename}"
            info = [
                html.P(f"Image: {filename}"),
                html.P(f"Ground Truth: {ground_truth}")
            ]
            if len(cd) >= 10:
                info.append(html.P(f"Predicted: {cd[3]}"))
                info.append(html.P(f"Top1: {cd[4]} ({cd[5]})"))
                info.append(html.P(f"Top2: {cd[6]} ({cd[7]})"))
                info.append(html.P(f"Top3: {cd[8]} ({cd[9]})"))
            return raw_img_url, gradcam_img_url, info
        return "", "", "Click a point to see details."

    print(f"[INFO] Starting Dash app on http://127.0.0.1:{args.dash_port}")
    app.run_server(debug=True, port=args.dash_port)

if __name__ == "__main__":
    main()
