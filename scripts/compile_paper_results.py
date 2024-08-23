import json
from pathlib import Path
from shutil import copyfile


def extract_scores(models, categories, score_key):
    """
    Extracts a selected score for each category across multiple models.

    Args:
        models (list): A list of dictionaries, where each dictionary contains category scores for a model.
        categories (list): A list of top-level categories to extract scores for.
        score_key (str): The specific score to extract (e.g., 'precision').

    Returns:
        dict: A dictionary with categories as keys and lists of scores for each model.
    """
    # Initialize the result dictionary with categories as keys and empty lists as values
    result = {category: [] for category in categories}

    # Iterate over each model's dictionary
    for model in models:
        # For each category, append the selected score to the result
        for category in categories:
            score = model.get(category, {}).get(score_key)
            result[category].append(score)

    return result

def save_textable(result, models, methods, categories, filename='./table.tex'):    

    import pandas as pd

    # Create a DataFrame from the result dictionary
    df = pd.DataFrame(result)

    # Add an index column for the model number
    mapping_models = {
        'mlp': 'MLP',
        'resnet': 'Patch-wise ResNet',
        'unet': 'U-Net',
    }
    df.index = [mapping_models[item] for item in models]

    # use math mode for strings containing underscore
    mapping_segment = {
        'accuracy': 'Accuracy',
        'dice': 'Dice',
        'iou': 'IoU',
        't_s': '$t_s$',
        't_mm': '$t_{mm}$'
    }
    mapping_per_class = {
        'bg': 'Background',
        'hwm': 'Healthy WM',
        'twm': 'Tumor WM',
        'gm': 'GM',
    }

    if all(c in mapping_per_class.keys() for c in categories):
        categories_tex = [mapping_per_class[item] for item in categories]
    elif all(c in mapping_segment.keys() for c in categories):
        categories_tex = [mapping_segment[item] for item in categories]
    else:
        raise Exception("Unknown categories")

    # Start writing the LaTeX content
    latex_content = r"""%
        \begin{table}[h!]
            \centering
            \caption{""" + " ".join(filename.split('.')[0].split('_')[1:]).title() + r"""}
            \begin{tabular}{ll""" + "c" * len(categories_tex) + r"""}
            \toprule
            Model & Input & """ + " & ".join(categories_tex) + r""" \\
            \midrule\midrule
        """

    # Add rows to the LaTeX content
    for i, (index, row) in enumerate(df.iterrows()):
        latex_content += f"    {index} & " + methods[i] + " & " + " & ".join(f"{row[col]:.4f}" if row[col] is not None else "-" for col in categories) + r" \\" + "\n"

    # Finalize the LaTeX content
    latex_content += r"""
            \bottomrule\bottomrule
            \end{tabular}
            \label{""" + filename.split('.')[0].replace('_', ':') + r"""}
        \end{table}
        """

    # Save to a .tex file
    with open(filename, 'w') as f:
        f.write(latex_content)

def save_texfigure(paths, labels, filename='fig_segment.tex'):
    from collections import defaultdict

    # Group paths by their figure number prefix
    grouped_paths = defaultdict(list)
    for path in paths:
        prefix = '-'.join(path.split('-')[:2])  # Extract the prefix (e.g., 'fig-0', 'fig-1')
        grouped_paths[prefix].append(path)

    # Sort the groups and their paths for consistent order
    sorted_groups = sorted(grouped_paths.items(), key=lambda x: x[0])
    sorted_groups = [el[1] for el in sorted_groups] # skip prefix
    sorted_groups = [el[1:]+[el[0]] for el in sorted_groups] # move gt to last entry

    # Start building the LaTeX content
    latex_content = "\\begin{figure}[h!]\n\\centering\n"

    # Determine the width for each minipage
    label_width = "0.1\\textwidth"  # Width for labels
    image_width = f"{0.86/len(sorted_groups):.2f}\\textwidth"  # Adjust the image width accordingly

    # Loop over each label with its corresponding group of images
    for i, label in enumerate(labels):
        # Create the minipage for the label
        latex_content += f"\\begin{{minipage}}[t]{{{label_width}}}\n"
        #latex_content += f"\\vfill\n"
        #latex_content += f"\\centering\n\\textbf{{{label}}}\n"
        #latex_content += f"\\vfill\n"
        latex_content += f"\\raisebox{{2.15\\height}}{{\\parbox{{\\textwidth}}{{\\raggedright \\textbf{{{label}}}}}}}\n"
        latex_content += "\\end{minipage}\n"
        latex_content += "\\hfill\n"

        # Add images associated with this label
        for images in sorted_groups:
            # Filter images related to the current label's row
            #relevant_images = [img for img in images if label in img]

            # Create the minipage for each relevant image
            img = images[i]
            latex_content += f"\\begin{{minipage}}[b]{{{image_width}}}\n\\centering\n"
            latex_content += f"\\includegraphics[width=\\textwidth]{{{img}}}\\\\\n"
            latex_content += "\\end{minipage}\n"
            latex_content += "\\vspace{-.075cm}\n"
        
        # Add a line break between rows to ensure the next label and images are on a new line
        latex_content += "\\\\[1em]\n"

    latex_content += "\\caption{Generated segmentation results.}\n"
    latex_content += "\\label{fig:segment}\n"
    latex_content += "\\end{figure}\n"

    # Write the LaTeX content to the specified file
    with open(filename, 'w') as f:
        f.write(latex_content)


if __name__ == '__main__':

    group_name = 'htgm_wo_bg_locvar8'
    run_list = []
    for fn in Path('./' + group_name).glob('config_*.json'):
        with open(fn, 'r') as f:
            cfg = json.load(f)
            if cfg['data_subfolder'].__contains__('raw') and cfg['levels'] <= 1:
                run_list.append([fn, cfg['model'], cfg['levels']])
    sorted_runs = sorted(run_list, key=lambda x: (-int(x[2]), x[1]))
    models = [el[1] for el in sorted_runs if el[1]]
    methods = ['MMFF' if el[2] == 0 else 'Lu-Chipman' for el in sorted_runs]

    # table per class score
    tables = []
    for el in sorted_runs:
        el = str(el[0]).replace('config', 'table')
        with open(el, 'r') as f:
            tab = json.load(f)
        tables.append(tab)
        score_key = 'f1-score'
    result = extract_scores(tables, categories=['bg', 'hwm', 'twm', 'gm'], score_key=score_key)
    save_textable(result, models, methods, categories=['bg', 'hwm', 'twm', 'gm'], filename=group_name+'/'+'tab_'+score_key+'_per_class.tex')

    # table semantic segmentation score
    metrics = []
    for el in sorted_runs:
        el = str(el[0]).replace('config', 'metrics')
        with open(el, 'r') as f:
            tab = json.load(f)
        metrics.append(tab)
    save_textable(metrics, models, methods, categories=['accuracy', 'dice', 't_s', 't_mm'], filename=group_name+'/'+'tab_semantic_segmentation_scores.tex')

    # image results
    mapping_labels = {
        'mlp': 'MLP',
        'resnet': 'ResNet',
        'unet': 'U-Net',
    }
    img_paths, labels = [], []
    for k, el in enumerate(sorted_runs):
        method = ['MMFF', 'LC'][el[2]]
        for i in range(4):
            for img_type in ['heatmap', 'img_mask', 'img_pred']:
                step_num = str(1105+i) if el[1] != 'resnet' else str(2201+i)
                tail = '_' +  str(i) + '_' +  img_type + '_test_' + step_num + '.png'
                fn = str(el[0]).replace('config_', '').replace('.json', tail).split('/')[-1]
                img_path = Path(group_name) / 'downloaded_images' / fn
                if img_path.exists():
                    if img_type == 'img_pred':
                        dst = Path(group_name) / ('fig-' + str(i) + '-' + method + '-' + el[1] + '.png')
                    elif img_type == 'heatmap':
                        dst = Path(group_name) / ('fig-' + str(i) + '-' + method + '-' + el[1] + '-heatmap.png')
                    elif img_type == 'img_mask' and k == 0:
                        dst = Path(group_name) / ('fig-' + str(i) + '-' + method + '-gt.png')
                    else:
                        continue
                    if img_type != 'heatmap' and dst.name not in img_paths: 
                        img_paths.append(dst.name)
                        if i==0 and not dst.name.__contains__('gt'): labels.append(mapping_labels[el[1]] + '\\newline ' + method)
                    copyfile(img_path, dst)
                else:
                    raise Exception('Could not find image file')
                
    save_texfigure(img_paths, labels+['GT\\newline'], filename=group_name+'/'+'fig_segment.tex')