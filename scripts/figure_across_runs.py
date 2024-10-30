
import re
import shutil
from pathlib import Path


if __name__ == "__main__":

    group_names = ['kfold3_absence', 'kfold3_rota', 'kfold3_flips', 'kfold3_rotaflips']
    input_type = 'LC'
    combination = input_type+'-unet'
    kfold_index = 2
    output_folder = Path('.') / 'scripts' / 'imgs_across_runs'
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(exist_ok=True)
    nested_fnames = [[] for _ in range(7)]

    for j, group_name in enumerate(group_names):
        if not Path('./' + group_name).exists():
            raise Exception(str('./' + group_name) + ' does not exist')

        run_list = []
        fname = Path('.') / group_name / ('fig_segment_%s.tex' % str(kfold_index))
        if not fname.exists():
            raise Exception(str(fname) + ' does not exist')
        
        with open(fname) as f:
            tex_lines = f.readlines()
        
        # extract image filnames from tex files
        fnames = re.findall(r'fig-\d+-'+combination+'\.png', ' '.join(tex_lines))
        fnames_gt = re.findall(r'fig-\d+-'+combination.split('-')[0]+'-gt.png', ' '.join(tex_lines))

        for i, fname in enumerate(fnames):
            new_fname = group_name + '_' + fname
            # copy images to new location
            shutil.copy(Path(group_name) / fname, output_folder / new_fname)
            # keep track of file paths
            nested_fnames[i].append(new_fname)

        # add GT at below all
        if j == len(group_names) - 1:
            for i, fname in enumerate(fnames_gt):
                new_fname = group_name + '_' + fname
                # copy images to new location
                shutil.copy(Path(group_name) / fname, output_folder / new_fname)
                # keep track of file paths
                nested_fnames[i].append(new_fname)

    import json
    for cfg_path in Path('./' + group_name).glob('config_*.json'):
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        if cfg['model'] == combination.split('-')[-1] and cfg['levels'] == 1:
            caption_name = 'captions_'+cfg_path.name.replace('json', 'yml').split('_')[-1]
            break

    # load image captions/labels
    import yaml
    with open(Path(group_name) / caption_name, 'r') as f:
        captions = list(yaml.safe_load(f).values())
    captions = [c.split(',')[0].replace('Astrocytoma','A').replace('Oligodendroglioma','O') + c.split('WHO')[-1].replace('grade:','') if c != 'healthy' else c for c in captions]

    # create new global tex figure
    from scripts.compile_paper_results import save_texfigure, compile_pdf
    labels = [term.split('_')[-1].capitalize() for term in  group_names]
    save_texfigure(nested_fnames, labels+['GT\\newline'], filename=output_folder / 'fig_across_runs.tex', captions=captions)
    compile_pdf(output_folder, 'fig_across_runs.tex', latex_file='figure_env.tex')
