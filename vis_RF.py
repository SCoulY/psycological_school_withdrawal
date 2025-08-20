from sklearn.tree import export_graphviz
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from preprocess.ch2en import column_name2eng
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
from typing import List
from shutil import which
import re


# print(plt.style.available)
# plt.rcParams['font.family'] = ['Noto Sans']

'''Empirical top 10 features from the analysis'''
top10_features_adults = ['HEI_TS', 'SCL-90 DEP', 'CSES_TS', 'SCL-90 PSY', 'SCL-90 ANX', 'EMBU-F EW', 'DES-Ⅱ_TS', 'DES-Ⅱ_ABS', 'DES-Ⅱ_AMN', 'EMBU-M FS']

top10_features_teens = ['SCL-90 DEP', 'A-DES-Ⅱ_PI', 'SCL-90 ANX', 'CSES_TS', 'EMBU-F EW', 'HEI_TS', 'SCL-90 PSY', 'A-DES-Ⅱ_DPDR', 'A-DES-Ⅱ_AII', 'EMBU-M FS']

top10_features_children = ['CSES_TS', 'HEI_TS', 'A-DES-Ⅱ_PI', 'A-DES-Ⅱ_AII', 'A-DES-Ⅱ_DPDR', 'EMBU-F EW', 'EMBU-F PUN', 'EMBU-M FS', 'CSQ_REP', 'A-SSRS_OS']

def export_tree_graphviz(estimator, feature_names: List[str], class_names: List[str], out_path: str, max_display_depth: int = 4,
                         left_to_right: bool = True, fontname: str = "Helvetica", fontsize: int = 10,
                         ranksep: float = 0.9, nodesep: float = 0.25, simplify_labels: bool = False):
    """Export a single sklearn tree to a compact Graphviz image.

    Parameters
    ----------
    estimator : DecisionTreeClassifier
        Individual tree from RandomForest.
    feature_names, class_names : list
        Names for features/classes.
    out_path : str
        Output filepath WITHOUT extension. (Graphviz adds .gv / image ext.)
    max_display_depth : int
        Prunes the *visualization only* to keep figure small.
    left_to_right : bool
        Orient depth horizontally (rankdir=LR) to reduce vertical height.
    """
    dot = export_graphviz(
        estimator,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        proportion=True,
        impurity=False,
        max_depth=max_display_depth,
        precision=2
    )
    if simplify_labels:
        # Remove samples / value lines to make nodes shorter
        dot = re.sub(r"samples = [^\\n]*\\n", "", dot)
        dot = re.sub(r"value = [^\\n]*\\n", "", dot)

    # Inject styling tweaks (graph-level spacing to reduce overlap)
    attrs = [
        "rankdir=LR" if left_to_right else "",
        f"graph [ranksep={ranksep}, nodesep={nodesep}]",
        f"node [shape=box style=rounded height=0.3 width=0.4 fontname=\"{fontname}\" fontsize={fontsize} penwidth=1]",
        "edge [penwidth=1]"
    ]
    inject = "\n".join([a for a in attrs if a])
    dot = dot.replace("digraph Tree {", f"digraph Tree {{\n{inject}\n")
    # Render to PNG & SVG for sharpness
    src = graphviz.Source(dot, filename=out_path)
    try:
        src.format = 'png'
        src.render(cleanup=True)
        src.format = 'svg'
        src.render(cleanup=True)
    except graphviz.backend.ExecutableNotFound as e:
        raise RuntimeError("Graphviz 'dot' executable not found. Install Graphviz and ensure 'dot' is on PATH or use --viz_mode matplotlib.\n"\
                           "Windows quick install: winget install --id Graphviz.Graphviz -e or download from https://graphviz.org/download/ and add its bin folder to PATH.") from e
    return out_path + '.png'


def visualize_three_trees_graphviz(estimators, tree_indices, feature_names, class_names, output_path, table_name,
                                   max_display_depth=4, left_to_right=True, ranksep=0.9, nodesep=0.25,
                                   simplify_labels=False):
    os.makedirs(output_path, exist_ok=True)
    image_paths = []
    for idx, est in zip(tree_indices, estimators):
        base = os.path.join(output_path, f"{table_name}_tree_{idx+1}")
    image_path = export_tree_graphviz(est, feature_names, class_names, base, max_display_depth, left_to_right,
                      ranksep=ranksep, nodesep=nodesep, simplify_labels=simplify_labels)
    image_paths.append(image_path)

    # Combine into a single side-by-side figure
    imgs = [plt.imread(p) for p in image_paths]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, img, idx in zip(axes, imgs, tree_indices):
        ax.imshow(img)
        ax.set_title(f'Tree #{idx+1}', fontsize=14)
        ax.axis('off')
    fig.suptitle(f'{table_name} – Representative Trees (pruned depth={max_display_depth})', fontsize=16)
    fig.tight_layout()
    combo_path = os.path.join(output_path, f'{table_name}_trees_{tree_indices[0]+1}_{tree_indices[1]+1}_{tree_indices[2]+1}_graphviz.pdf')
    fig.savefig(combo_path, bbox_inches='tight', format='pdf')
    return combo_path


def visualize_three_trees_matplotlib(estimators, tree_indices, feature_names, class_names, output_path, table_name,
                                     max_display_depth=None, tree_figsize=(8,4), simplify_labels=False):
    os.makedirs(output_path, exist_ok=True)
    sns.set_theme(style="white", font_scale=1.0)
    # Adjust: horizontal layout reduces overall height
    # Derive overall figure size based on single-tree figsize to spread nodes
    tw, th = tree_figsize
    fig, axes = plt.subplots(1, 3, figsize=(3*tw+6, th+4))
    for ax, est, idx in zip(axes, estimators, tree_indices):
        plot_tree(est, feature_names=feature_names, class_names=class_names, filled=True, rounded=True,
                  fontsize=8, ax=ax, max_depth=max_display_depth, impurity=False, proportion=True)
        if simplify_labels:
            # Remove text lines we don't want by clearing and re-drawing shorter labels isn't trivial with plot_tree;
            # Instead, we can shrink font or leave as-is. Placeholder for advanced custom drawing.
            pass
        ax.set_title(f'Tree #{idx+1}', fontsize=12)
    fig.suptitle(f'{table_name} – Representative Trees', fontsize=16)
    fig.tight_layout()
    out_file = os.path.join(output_path, f'{table_name}_trees_{tree_indices[0]+1}_{tree_indices[1]+1}_{tree_indices[2]+1}_matplotlib.pdf')
    fig.savefig(out_file, bbox_inches='tight', format='pdf')
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--output_path', type=str, default='C:/Users/SCoulY/Desktop/psycology/data/test_RF_confidence')
    parser.add_argument('--disable_top10', default=True, action='store_true', help='Use all features (default). Pass --no-disable_top10 to enable top10.')
    parser.add_argument('--viz_mode', choices=['graphviz', 'matplotlib'], default='graphviz', help='Visualization backend.')
    parser.add_argument('--display_max_depth', type=int, default=4, help='Max depth for display (graphviz or matplotlib).')
    parser.add_argument('--no_left_to_right', action='store_true', help='If set, keep top-to-bottom orientation for graphviz.')
    parser.add_argument('--ranksep', type=float, default=0.9, help='Graphviz rank separation (increase to spread levels).')
    parser.add_argument('--nodesep', type=float, default=0.25, help='Graphviz node separation (increase to add horizontal space).')
    parser.add_argument('--simplify_labels', action='store_true', help='Remove samples/value lines to reduce node height.')
    parser.add_argument('--tree_fig_width', type=float, default=8.0, help='Per-tree figure width for matplotlib backend.')
    parser.add_argument('--tree_fig_height', type=float, default=4.0, help='Per-tree figure height for matplotlib backend.')
    # Multi-cohort specific arguments (all optional; if all three provided we enter multi-cohort mode)
    parser.add_argument('--file_path_adults', type=str, help='Adults cohort data CSV', default='C:/Users/SCoulY/Desktop/psycology/data/clean_adults.csv')
    parser.add_argument('--file_path_teens', type=str, help='Teens cohort data CSV', default='C:/Users/SCoulY/Desktop/psycology/data/clean_teens.csv')
    parser.add_argument('--file_path_children', type=str, help='Children cohort data CSV', default='C:/Users/SCoulY/Desktop/psycology/data/clean_teens_wo_scl.csv')
    parser.add_argument('--ckpt_path_adults', type=str, help='Adults RF model .pkl', default='C:/Users/SCoulY/Desktop/psycology/ckpt_5runs/children_correct/full/adults/clean_adults_RandomForest_acc_0.90_run_255.pkl')
    parser.add_argument('--ckpt_path_teens', type=str, help='Teens RF model .pkl', default='C:/Users/SCoulY/Desktop/psycology/ckpt_5runs/children_correct/full/teens/clean_teens_RandomForest_acc_0.82_run_6.pkl')
    parser.add_argument('--ckpt_path_children', type=str, help='Children RF model .pkl', default='C:/Users/SCoulY/Desktop/psycology/ckpt_5runs/children_correct/full/children/clean_teens_wo_scl_RandomForest_acc_0.72_run_6.pkl')
    args = parser.parse_args()

    def load_dataset_and_model(file_path, ckpt_path, cohort_name_hint=None):
        rf_model_local = joblib.load(ckpt_path)
        df_local = pd.read_csv(file_path)
        df_local = df_local.drop(df_local.columns[0], axis=1)
        df_local = column_name2eng(df_local)
        if 'School Withdrawal/ Reentry Status' in df_local.columns:
            df_local = df_local.drop(columns=['School Withdrawal/ Reentry Status'])

        # Determine cohort label / top10 features
        if cohort_name_hint:
            hint = cohort_name_hint.lower()
        else:
            hint = file_path.lower()
        if 'adult' in hint:
            cohort = 'Adults'
            top10 = top10_features_adults
        elif 'teen' in hint and 'wo_scl' not in hint:
            cohort = 'Teens'
            top10 = top10_features_teens
        elif 'child' in hint or 'wo_scl' in hint:
            cohort = 'Children'
            top10 = top10_features_children
        else:
            cohort = 'Dataset'
            top10 = df_local.columns.tolist()

        feature_names_local = top10 if not args.disable_top10 else df_local.columns.tolist()
        feature_names_local = [f.replace('Ⅱ', 'II') for f in feature_names_local]
        return rf_model_local, df_local, feature_names_local, cohort

    multi_mode = all([
        args.file_path_adults, args.ckpt_path_adults,
        args.file_path_teens, args.ckpt_path_teens,
        args.file_path_children, args.ckpt_path_children
    ])

    tree_indices = [0, 49, 99]
    class_names = ['withdrawal', 'reentry']

    def _graphviz_available():
        return which('dot') is not None

    if args.viz_mode == 'graphviz' and not _graphviz_available():
        print("[WARN] Graphviz 'dot' not found on PATH. Falling back to matplotlib. Install Graphviz or rerun with --viz_mode matplotlib explicitly.")
        args.viz_mode = 'matplotlib'

    if not multi_mode:
        # Single cohort legacy behavior
        rf_model, df, feature_names, table_name = load_dataset_and_model(args.file_path, args.ckpt_path)
        estimators = [rf_model.estimators_[i] for i in tree_indices]
        if args.viz_mode == 'graphviz':
            try:
                                visualize_three_trees_graphviz(estimators, tree_indices, feature_names, class_names, args.output_path,
                                                                                             table_name, max_display_depth=args.display_max_depth,
                                                                                             left_to_right=not args.no_left_to_right,
                                                                                             ranksep=args.ranksep, nodesep=args.nodesep,
                                                                                             simplify_labels=args.simplify_labels)
            except RuntimeError as e:
                print(f"[ERROR] {e}\nFalling back to matplotlib visualization.")
                visualize_three_trees_matplotlib(estimators, tree_indices, feature_names, class_names, args.output_path,
                                                                                                 table_name, max_display_depth=args.display_max_depth,
                                                                                                 tree_figsize=(args.tree_fig_width, args.tree_fig_height),
                                                                                                 simplify_labels=args.simplify_labels)
        else:
            visualize_three_trees_matplotlib(estimators, tree_indices, feature_names, class_names, args.output_path,
                                                                                         table_name, max_display_depth=args.display_max_depth,
                                                                                         tree_figsize=(args.tree_fig_width, args.tree_fig_height),
                                                                                         simplify_labels=args.simplify_labels)
        print(f'Finished visualizing trees {tree_indices} with mode={args.viz_mode}. Output folder: {args.output_path}')
    else:
        # Multi-cohort mode: Adults, Teens, Children
        cohorts_input = [
            ('Adults', args.file_path_adults, args.ckpt_path_adults),
            ('Teens', args.file_path_teens, args.ckpt_path_teens),
            ('Children', args.file_path_children, args.ckpt_path_children)
        ]
        # Build grid: 3 large subplots, one per cohort; each combines 3 tree images
        combined_images = []  # list of (cohort_name, composite_array)
        for cohort_hint, fp, cp in cohorts_input:
            rf_model, df, feature_names, table_name = load_dataset_and_model(fp, cp, cohort_name_hint=cohort_hint)
            estimators = [rf_model.estimators_[i] for i in tree_indices]
            # Generate individual tree images (png) using chosen backend
            if args.viz_mode == 'graphviz':
                try:
                    image_paths = []
                    for idx in tree_indices:
                        base = os.path.join(args.output_path, f"{table_name}_tree_{idx+1}")
                        image_paths.append(export_tree_graphviz(rf_model.estimators_[idx], feature_names, class_names, base,
                                                                max_display_depth=args.display_max_depth,
                                                                left_to_right=not args.no_left_to_right,
                                                                ranksep=args.ranksep, nodesep=args.nodesep,
                                                                simplify_labels=args.simplify_labels))
                except RuntimeError as e:
                    print(f"[ERROR] {e}\nFalling back to matplotlib visualization for cohort {table_name}.")
                    args.viz_mode = 'matplotlib'
            if args.viz_mode == 'matplotlib':
                image_paths = []
                # Create temporary figure with 3 trees to then split? Easier: draw each tree separately
                for idx in tree_indices:
                    fig_tmp, ax_tmp = plt.subplots(figsize=(args.tree_fig_width, args.tree_fig_height))
                    plot_tree(rf_model.estimators_[idx], feature_names=feature_names, class_names=class_names,
                              filled=True, rounded=True, fontsize=6, ax=ax_tmp, max_depth=args.display_max_depth,
                              impurity=False, proportion=True)
                    if args.simplify_labels:
                        pass
                    ax_tmp.set_title(f'Tree #{idx+1}', fontsize=8)
                    fig_path = os.path.join(args.output_path, f"{table_name}_tree_{idx+1}_matplotlib.png")
                    fig_tmp.savefig(fig_path, dpi=200, bbox_inches='tight')
                    plt.close(fig_tmp)
                    image_paths.append(fig_path)
            # Compose horizontal strip
            imgs = [plt.imread(p) for p in image_paths]
            heights = [im.shape[0] for im in imgs]
            max_h = max(heights)
            padded = []
            for im in imgs:
                if im.shape[0] < max_h:
                    pad_h = max_h - im.shape[0]
                    pad = np.ones((pad_h, im.shape[1], im.shape[2]))  # white
                    pad[:] = 1.0
                    im = np.vstack([im, pad])
                padded.append(im)
            composite = np.hstack(padded)
            combined_images.append((table_name, composite))

        # Plot combined figure
        fig, axes = plt.subplots(len(combined_images), 1, figsize=(35, 7 * len(combined_images)))
        if len(combined_images) == 1:
            axes = [axes]
    panel_letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    for i, (ax, (cohort_name, comp_img)) in enumerate(zip(axes, combined_images)):
        ax.imshow(comp_img, aspect='auto')
        ax.axis('off')
        # Cohort name centered above image
        ax.set_title(cohort_name, fontsize=18, pad=12)
        # Panel letter placed using axes coordinates for consistent alignment
        ax.text(0, 1.02, f'({panel_letters[i]})', transform=ax.transAxes,
            fontsize=20, fontweight='bold', ha='right', va='bottom')
    fig.suptitle('Representative Trees (indices 1, 50, 100) Across Cohorts', fontsize=24, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    multi_out = os.path.join(args.output_path, f'AllCohorts_trees_1_50_100_{args.viz_mode}.pdf')
    fig.savefig(multi_out, bbox_inches='tight', format='pdf')
    print(f'Multi-cohort visualization saved to {multi_out}')