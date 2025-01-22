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

def build_figure_for_l2(df_all, l2_word_label, image_port):
    """
    Given the full DataFrame (df_full), filter rows for the specified L2 label,
    then build and return a Plotly scatter figure with color-coded word labels
    and custom shapes.
    """

    # If l2_label is None or "ALL", skip filtering. Otherwise, filter the DataFrame
    if l2_word_label is None or l2_word_label == "ALL":
        df_filtered = df_all.copy()
    else:
        # Convert the chosen L2 word label to numeric
        # e.g. "DOG_FAMILY" -> 1
        selected_l2_label = NAME_LABEL_L2.get(l2_word_label)
        if selected_l2_label is None:
            return px.scatter()  # Return empty figure if unrecognized

        # Build a set of L3 numeric IDs that belong to that L2
        l3_ids = []
        for l3_word, (l3_label, parent_l2_label) in REASSIGN_NAME_LABEL_L3L2.items():
            if parent_l2_label == selected_l2_label:
                l3_ids.append(l3_label)

        # Filter to only those L3 IDs
        df_filtered = df_all[df_all['label'].isin(l3_ids)]

    # Build the figure. We'll:
    # - color by L3 word_label
    # - use symbol by L3 word_label (for varied shapes)
    # - have hover_data show filename, word_label
    # - store img_url in custom_data so we can retrieve the image on click
    fig = px.scatter(
        df_filtered,
        x="tsne_x",
        y="tsne_y",
        color="word_label",         # color by the word label
        symbol="word_label",        # shape by word label
        hover_data=["filename", "word_label"],
        custom_data=["img_url"],    # for retrieving image on click
        # You could also adjust the size if you have a column for that
        # size="some_size_column",
    )

    # Tweak layout
    fig.update_layout(
        title=f"t-SNE (L2: {l2_word_label}) - {len(df_filtered)} points" if l2_word_label else "t-SNE (All data)",
        legend_title="L3 Word Label"
    )
    return fig

def filter_df(df_all, l2_word_label):
    """
    Selecting data points with l2_cls
    :param df_all:
    :param l2_word_label:
    :return:
    """
    if l2_word_label not in NAME_LABEL_L2:
        raise ValueError(f"{l2_word_label} not recognized for L2 class.")
    selected_l2_label = NAME_LABEL_L2[l2_word_label]

    # Build a set of L3 numeric IDs that belong to that L2
    l3_ids = []
    for l3_word_label, (l3_label, l2_label) in REASSIGN_NAME_LABEL_L3L2.items():
        if l2_label == selected_l2_label:
            l3_ids.append(l3_label)

    df = df_all[df_all['label'].isin(l3_ids)]
    print(f"[INFO] Filtered to L2 label '{l2_word_label}' with {len(df)} samples.")
    return df

def main():
    # 1) PARSE COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser(
        description="t-SNE Viewer with local image hosting using Dash."
    )
    parser.add_argument(
        "--tsne-results",
        type=str,
        default="tsne_results.csv",
        help="Path to the CSV file containing t-SNE results (columns: tsne_x, tsne_y, label, filename)."
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
        help="Name of L2 Class (e.g., Grassland).")

    args = parser.parse_args()

    tsne_csv = os.path.abspath(args.tsne_results)
    image_server_port = args.image_port
    dash_port = args.dash_port

    # ----------------------------------------------------------------
    # 2) START THE LOCAL IMAGE SERVER ON A BACKGROUND THREAD
    # ----------------------------------------------------------------
    # We'll start a separate thread so that the Dash app can run concurrently
    # in the main thread.
    # image_folder = os.path.abspath(args.image_folder)
    # server_thread = threading.Thread(
    #     target=serve_folder,
    #     args=(image_folder, image_server_port),
    #     daemon=True
    # )
    # server_thread.start()

    # 2) LOAD T-SNE RESULTS (CSV)
    if not os.path.exists(tsne_csv):
        raise FileNotFoundError(f"[ERROR] t-SNE results file not found: {tsne_csv}")

    df = pd.read_csv(tsne_csv)

    required_cols = {"tsne_x", "tsne_y", "label", "filename"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # 3) Create a 'word_label' column from numeric L3 'label'
    df["word_label"] = df["label"].map(REASSIGN_LABEL_NAME_L3)

    # 4) Create an 'img_url' column for the samples
    # Build the image URL from your local server
    # e.g. http://localhost:8888/ + filename
    # base_url = f"http://127.0.0.1:{image_server_port}/"
    base_url = f"http://localhost:{image_server_port}/"
    df["img_url"] = base_url + df["filename"].astype(str)

    # 5) Create the Dash app
    app = Dash(__name__)

    # We'll store the entire df in a dcc.Store so the callback can filter it.
    # Alternatively, we could just keep it in a global var, but let's be "clean."
    app.layout = html.Div([
        dcc.Store(id="full-data", data=df.to_dict(orient="records")),

        html.H3("Select L2 Class:"),
        dcc.Dropdown(
            id="l2-dropdown",
            options=[{"label": "ALL", "value": "ALL"}] + [
                {"label": l2_name, "value": l2_name}
                for l2_name in NAME_LABEL_L2.keys()
            ],
            value="ALL",  # default to showing all
            clearable=False,
            style={"width": "300px"}
        ),

        # Graph
        dcc.Graph(
            id="tsne-graph",
            style={
                "width": "70%",
                "height": "800px",  # <-- Make the plot bigger/taller
                "display": "inline-block"
            }
        ),

        # Image on the side
        html.Div(
            [
                html.Img(
                    id="clicked-image",
                    style={
                        "width": "300px",
                        "height": "auto",
                        "border": "1px solid #ccc",
                        "padding": "5px"
                    }
                )
            ],
            style={"display": "inline-block", "verticalAlign": "top", "marginLeft": "20px"}
        )
    ])

    # 6) Callback to update the scatter figure whenever the L2 dropdown changes
    @app.callback(
        Output("tsne-graph", "figure"),
        [Input("l2-dropdown", "value")],
        [State("full-data", "data")]
    )
    def update_figure(l2_value, stored_data):
        # Reconstruct a DataFrame from the stored data
        df_full = pd.DataFrame(stored_data)
        # Build the figure by filtering on l2_value
        fig = build_figure_for_l2(df_full, l2_value, args.image_port)
        return fig

    # 7) Callback to update the displayed image upon clicking a point
    @app.callback(
        Output("clicked-image", "src"),
        [Input("tsne-graph", "clickData")]
    )
    def update_image(clickData):
        if clickData and "points" in clickData:
            # We only allow single selection, so take the first clicked point
            point = clickData["points"][0]
            img_url = point["customdata"][0]  # custom_data = ["img_url"]
            return img_url
        return None

    # 8) Run
    print(f"[INFO] Starting Dash app on http://127.0.0.1:{dash_port}")
    app.run_server(debug=True, port=dash_port)


if __name__ == "__main__":
    main()
