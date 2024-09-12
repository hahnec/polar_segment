import os
import json
import wandb
import requests
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from io import BytesIO
import imageio


def convert_to_markdown(table):
    headers = table.columns
    rows = table.data

    # Construct the markdown table
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for row in rows:
        row = [str(cell) if cell is not None else '' for cell in row]
        markdown += "| " + " | ".join(map(str, row)) + " |\n"

    return markdown

def dict_to_markdown_table(d, digits=4):
    # Keys and values
    keys = list(d.keys())
    values = list(d.values())

    # Construct the headers and rows
    headers = "| " + " | ".join(keys) + " |"
    separator = "| " + " | ".join(["---"] * len(keys)) + " |"
    value_row = "| " + " | ".join(map(str, [round(el, digits) for el in values if el is not None])) + " |"

    # Combine everything into the markdown table
    markdown_table = f"{headers}\n{separator}\n{value_row}"
    
    return markdown_table

def download_image(url, save_path):
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        pil_object = Image.open(BytesIO(response.content))
        imageio.imsave(save_path, np.array(pil_object))
    elif response.status_code == 404:
        raise Exception('Page does not exist')


if __name__ == "__main__":

    # load config
    cfg = OmegaConf.load('./configs/keys.yml')

    # Log in to wandb
    wandb.login(key=cfg.wandb_api)

    # Define the project and group name
    group_name = 'kfold_200epochs_balance'
    project_name = 'polar_segment'  
    table_key = 'report'
    media_keys = ['heatmap_test', 'img_pred_test', 'img_mask_test']

    api = wandb.Api()

    runs = api.runs(path=f'{wandb.Api().default_entity}/{project_name}', filters={"group": group_name})

    output_dir = os.path.join(group_name, 'downloaded_images')
    os.makedirs(output_dir, exist_ok=True)

    for run in runs:

        if run.state != 'finished':
            print(run.name + ' ' + run.state)
            continue
        
        # config
        with open(os.path.join(group_name, 'config_%s.json' % str(run.name)), 'w') as f:
            json.dump(run.config, f, indent=4)

        metrics = {}
        for col in ['accuracy', 'dice', 'iou', 'auc', 't_mm', 't_s']:
            metrics[col] = run.summary.get(col)
        with open(os.path.join(group_name, 'metrics_%s.json' % str(run.name)), 'w') as f:
            json.dump(metrics, f, indent=4)
        #md_metrics = dict_to_markdown_table(metrics)
        #with open(os.path.join(group_name, 'metrics_%s.md' % str(run.name)), 'w') as f:
        #    f.write(md_metrics)

        table = run.use_artifact(run.logged_artifacts()[1]).get(table_key)
        table_dict = {
            row[0]: {table.columns[i]: row[i] for i in range(1, len(table.columns))}
            for row in table.data
        }
        with open(os.path.join(group_name, 'table_%s.json' % str(run.name)), 'w') as f:
            json.dump(table_dict, f, indent=4)
        md_table = convert_to_markdown(table)
        with open(os.path.join(group_name, 'table_%s.md' % str(run.name)), 'w') as f:
            f.write(md_table)

        # images
        captions = {}
        base_url = 'https://api.wandb.ai/files/hahnec/' + project_name + '/' + run.url.split('runs/')[-1] + '/'
        for media_key in media_keys:
            media_files = run.history(keys=[media_key]).get(media_key, [])
            for i, media in enumerate(media_files):
                if isinstance(media, dict) and 'path' in media:
                    image_url = base_url + media['path']
                    base_name = '_'.join(os.path.basename(image_url).split('_')[:-1])+'.png' # skip hash clutter
                    image_name = f"{run.name}_{i}_{base_name}"
                    save_path = os.path.join(output_dir, image_name)
                    download_image(image_url, save_path)
                    captions[str(i)] = media['caption'] # store captions containing labels
        import yaml
        with open(os.path.join(group_name, ('captions_'+run.name+'.yml')), 'w') as f:
            yaml.dump(captions, f)

        # download ROC plot data
        artifact_dict = run.history(keys=['roc_table']).get('roc_table', [])[0]
        roc_url = base_url + artifact_dict['path']
        response = requests.get(roc_url)
        if response.status_code == 200:
            content_str = response.content.decode('utf-8')
            roc_dict = json.loads(content_str)
            with open(os.path.join(group_name, 'roc_%s.json' % str(run.name)), 'w') as f:
                json.dump(roc_dict, f)

        # download training and validation curves
        curves_dict = {
            'train_loss': None, 'valid_loss': None, 
            'train_dice': None, 'valid_dice': None,
            'train_acc': None, 'valid_acc': None,
            }
        for k in curves_dict.keys():
            curves_dict[k] = [r[k] for _, r in run.history(keys=[k]).iterrows()]
        with open(os.path.join(group_name, 'curves_%s.json' % str(run.name)), 'w') as f:
            json.dump(curves_dict, f)

    print("Done.")
