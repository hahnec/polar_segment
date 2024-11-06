import json
from pathlib import Path
from shutil import copyfile
import subprocess
import os

def compile_latex_to_pdf(latex_file):
    # Extract the file name without the extension
    output_dir = os.path.dirname(latex_file)
    file_name = os.path.basename(latex_file)
    name_without_ext = os.path.splitext(file_name)[0]
    
    # Run pdflatex to compile the LaTeX file
    subprocess.run(["pdflatex", "-output-directory", output_dir, latex_file])
    
    # The PDF will be saved in the same directory as the LaTeX file
    pdf_path = os.path.join(output_dir, name_without_ext + ".pdf")
    return pdf_path

def compile_pdf(group_name, fig_texname, latex_file="./figure_env.tex"):

    group_name = str(group_name)

    # standalone environment
    latex_env = "\\documentclass{scrartcl}\n"
    latex_env += "\\usepackage[margin=0.5in]{geometry}\n"
    latex_env += "\\usepackage{graphicx}\n"
    latex_env += "\\usepackage{booktabs}\n"
    latex_env += "\\graphicspath{{%s/}}\n" % group_name
    latex_env += "\\begin{document}\n"
    latex_env += "\\input{%s/%s}\n" % (group_name, fig_texname)
    latex_env += "\\end{document}\n"

    with open(group_name+'/'+latex_file, 'w') as f:
        f.write(latex_env)

    pdf_output = compile_latex_to_pdf(group_name+'/'+latex_file)
    print(f"PDF generated at: {pdf_output}")

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

def save_textable(result, models, methods, categories, filename='./table.tex', digits=4):    

    import pandas as pd

    # Create a DataFrame from the result dictionary
    df = pd.DataFrame(result)

    # Add an index column for the model number
    mapping_models = {
        'mlp': 'MLP',
        'resnet': 'ResNet',
        'unet': 'U-Net',
    }
    df.index = [mapping_models[item] for item in models]

    # use math mode for strings containing underscore
    mapping_segment = {
        'accuracy': 'Accuracy',
        'dice': 'DSC',
        'auc': 'AUC',
        't_s': '$t_s~[s]$',
        't_mm': '$t_{mm}~[s]$'
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
        latex_content += f"     {index} & " + methods[i] + " & " + " & ".join("-" if row[col] is None else f"{row[col]:.{digits}f}" if isinstance(row[col], float) else f"{row[col][0]:.{digits}f} $\pm$ {row[col][1]:.{digits}f}" for col in categories) + r" \\" + "\n"

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

def save_texfigure(sorted_groups, labels, filename='fig_segment.tex', captions=None):

    # Start building the LaTeX content
    latex_content = "\\begin{figure}[h!]\n\\centering\n"

    # Determine the width for each minipage
    label_width = "0.1\\textwidth"  # Width for labels
    image_width = f"{0.825/len(sorted_groups):.2f}\\textwidth"  # Adjust the image width accordingly

    # Loop over each label with its corresponding group of images
    for i, label in enumerate(labels):
        # Create the minipage for the label
        latex_content += f"\\begin{{minipage}}[t]{{{label_width}}}\n"
        latex_content += f"\\raisebox{{1.15\\height}}{{\\parbox{{\\textwidth}}{{\\raggedright \\textbf{{{label}}}}}}}\n"
        latex_content += "\\end{minipage}\n"
        latex_content += "\\hfill\n"

        # Add images associated with this label
        for k, images in enumerate(sorted_groups):
            # Create the minipage for each relevant image
            img = images[i]
            latex_content += f"\\begin{{minipage}}[b]{{{image_width}}}\n\\centering\n"
            latex_content += f"\\includegraphics[width=\\textwidth]{{{img}}}\\\\\n"
            if i+1 == len(labels): latex_content += f"{captions[k]}\n"
            latex_content += "\\end{minipage}\n"
            #latex_content += "\\vspace{-.075cm}\n"
        
        # Add a line break between rows to ensure the next label and images are on a new line
        latex_content += "\\\\[1em]\n"

    latex_content += "\\caption{Generated segmentation results.}\n"
    latex_content += "\\label{fig:segment}\n"
    latex_content += "\\end{figure}\n"

    # Write the LaTeX content to the specified file
    with open(filename, 'w') as f:
        f.write(latex_content)

def merge_kfold_score(result, models, methods):

    new_models = []
    new_methods = []
    merge_dict = {k: {} for k in result} if isinstance(result, dict) else {}
    new_results = {k: [] for k in result} if isinstance(result, dict) else {}
    for j, c in enumerate(result):
        if not any(result[c]):
            merge_dict.pop(c)
            new_results.pop(c)
            continue
        for i, (model, method) in enumerate(zip(models, methods)):
            if not model+'_'+method in merge_dict[c].keys(): merge_dict[c][model+'_'+method] = []
            merge_dict[c][model+'_'+method] = list(merge_dict[c][model+'_'+method]) + [result[c][i]]
        
        for k in merge_dict[c].keys():
            data = merge_dict[c][k]
            mean = sum(data) / len(data)
            vari = sum((x - mean) ** 2 for x in data) / len(data)
            std = vari**.5
            new_results[c].append((mean, std))
            
            if j == 1:  # do once only for index one in case background is empty
                model, method = k.split('_')
                new_models.append(model)
                new_methods.append(method)

    return new_results, new_models, new_methods


if __name__ == '__main__':

    group_name = 'kfold3_absence'
    kfold_opt = group_name.lower().translate(str.maketrans('', '', '-_ ')).__contains__('kfold')
    kfold_num = int(group_name.lower().translate(str.maketrans('', '', '-_ ')).split('kfold')[-1][0])
    run_list = []
    for fn in Path('./' + group_name).glob('config_*.json'):
        with open(fn, 'r') as f:
            cfg = json.load(f)
            if cfg['data_subfolder'].__contains__('raw') and cfg['levels'] <= 1:
                run_list.append([fn, cfg['model'], cfg['levels']])
    if len(run_list) == 0:
        raise Exception('Empty folder')
    sorted_runs = sorted(run_list, key=lambda x: (-int(x[2]), x[1], int(str(x[0].name).split('-')[-1].split('.')[0])))
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
    n_result, n_models, n_methods = merge_kfold_score(result, models, methods) if kfold_opt else (result, models, methods)
    save_textable(n_result, n_models, n_methods, categories=list(n_result.keys()), filename=group_name+'/'+'tab_'+score_key+'_per_class.tex')

    # table semantic segmentation score
    metrics = []
    for el in sorted_runs:
        el = str(el[0]).replace('config', 'metrics')
        with open(el, 'r') as f:
            tab = json.load(f)
        metrics.append(tab)
    metrics = {key: [d[key] for d in metrics] for key in metrics[0]}
    n_metrics, n_models, n_methods = merge_kfold_score(metrics, models, methods) if kfold_opt else (metrics, models, methods)
    save_textable(n_metrics, n_models, n_methods, categories=['accuracy', 'dice', 'auc', 't_s', 't_mm'], filename=group_name+'/'+'tab_semantic_segmentation_scores.tex', digits=3)

    # image results
    mapping_labels = {
        'mlp': 'MLP',
        'resnet': 'ResNet',
        'unet': 'U-Net',
    }
    img_paths, labels = [], []
    e = 2
    img_columns, s = (7, 1000) if cfg['imbalance'] else (6, 900)
    img_columns = 10 if group_name.__contains__('test') else img_columns
    ks = kfold_num if kfold_opt else 1
    for k in range(ks):
        for j, el in enumerate(sorted_runs[k::ks]):
            method = ['MMFF', 'LC'][el[2]]
            for i in range(img_columns):
                for img_type in ['heatmap', 'img_mask', 'img_pred']:
                    step_num = str(s*e+5+i) if el[1] != 'resnet' else str(s*e+1+i)
                    tail = '_' +  str(i) + '_' +  img_type + '_test_' + step_num + '.png'
                    fn = str(el[0]).replace('config_', '').replace('.json', tail).split('/')[-1]
                    # automatically find filename with correct step number
                    img_path = list((Path(group_name) / 'downloaded_images').glob('_'.join(fn.split('_')[:-1])+'_*.png'))[0]
                    if img_path.exists():
                        if img_type == 'img_pred':
                            dst = Path(group_name) / ('fig-' + str(i) + '-' + method + '-' + el[1] + '.png')
                        elif img_type == 'heatmap':
                            dst = Path(group_name) / ('fig-' + str(i) + '-' + method + '-' + el[1] + '-heatmap.png')
                        elif img_type == 'img_mask' and j == 0:
                            dst = Path(group_name) / ('fig-' + str(i) + '-' + method + '-gt.png')
                        else:
                            continue
                        if img_type != 'heatmap' and dst.name not in img_paths: 
                            img_paths.append(dst.name)
                            if i==0 and not dst.name.__contains__('gt'): labels.append(mapping_labels[el[1]] + '\\newline ' + method)
                        copyfile(img_path, dst)
                    else:
                        raise Exception('Could not find image file')

        # load image captions/labels
        import yaml
        with open(Path(group_name) / ('captions_'+el[0].name.replace('json', 'yml').split('_')[-1]), 'r') as f:
            captions = list(yaml.safe_load(f).values())
        captions = [c.split(',')[0].replace('Astrocytoma','A').replace('Oligodendroglioma','O') + c.split('WHO')[-1].replace('grade:','') if c != 'healthy' else c for c in captions]

        # Group paths by their figure number prefix
        from collections import defaultdict
        grouped_paths = defaultdict(list)
        for path in img_paths:
            prefix = '-'.join(path.split('-')[:2])  # Extract the prefix (e.g., 'fig-0', 'fig-1')
            grouped_paths[prefix].append(path)

        # Sort the groups and their paths for consistent order
        sorted_groups = sorted(grouped_paths.items(), key=lambda x: x[0])
        sorted_groups = [el[1] for el in sorted_groups] # skip prefix
        sorted_groups = [el[1:]+[el[0]] for el in sorted_groups] # move gt to last entry

        # save figure tex file
        fig_texname = 'fig_segment_%s.tex' % str(k)
        save_texfigure(sorted_groups, labels+['GT\\newline'], filename=group_name+'/'+fig_texname, captions=captions)
        compile_pdf(group_name, fig_texname, latex_file='figure_env_'+str(k)+'.tex')

    # save table results (k-folds accumulated)
    compile_pdf(group_name, 'tab_semantic_segmentation_scores.tex', latex_file='/table_env.tex')
