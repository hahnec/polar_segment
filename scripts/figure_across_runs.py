
import re
import shutil
from pathlib import Path


if __name__ == "__main__":

    kfold_index = 2
    group_names = ['tip_mm_bs2_b_absence', 'tip_mm_bs2_b_noise', 'tip_mm_bs2_b_false_flip', 'tip_mm_bs2_b_false_rota', 'tip_mm_bs2_b_false_rotaflip', 'tip_mm_bs2_b_flip', 'tip_mm_bs2_b_rota', 'tip_mm_bs2_b_rotaflip']
    input_type = 'MMFF'
    combination = input_type+'-unet'
    output_folder = Path('.') / 'scripts' / 'imgs_across_runs'
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(exist_ok=True)

    # load captions from file
    import json
    for cfg_path in Path('./' + group_names[-1]).glob('config_*.json'):
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        if cfg['model'] == combination.split('-')[-1] and cfg['levels'] == 0 and kfold_index == cfg['k_select']:
            caption_name = 'captions_'+cfg_path.name.replace('json', 'yml').split('_')[-1]
            break

    # amend image captions/labels
    import yaml
    with open(Path(group_names[-1]) / caption_name, 'r') as f:
        captions = list(yaml.safe_load(f).values())
    captions = [c.split(',')[0].replace('Astrocytoma','A').replace('Oligodendroglioma','O') + c.split('WHO')[-1].replace('grade:','') if c != 'healthy' else c for c in captions]

    nested_fnames = [[] for _ in range(len(captions))]

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
        fnames = re.findall(r'fig-'+'\d+-'+combination+'\.png', ' '.join(tex_lines))
        fnames_gt = re.findall(r'fig-'+'\d+-'+combination.replace('LC', input_type).split('-')[0]+'-gt.png', ' '.join(tex_lines))

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

    # create new global tex figure
    from scripts.compile_paper_results import save_texfigure, compile_pdf
    labels = [term.split('_')[-1].capitalize() for term in  group_names]
    save_texfigure(nested_fnames, labels+['GT\\newline'], filename=output_folder / 'fig_across_runs.tex', captions=captions)
    compile_pdf(output_folder, 'fig_across_runs.tex', latex_file='figure_env.tex')
