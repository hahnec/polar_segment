import os
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
    project_name = 'polar_segment'
    group_name = '4_class_multi_viable_test_wo_bg'
    table_key = 'report'
    media_keys = ['heatmap_test', 'img_pred_test', 'img_mask_test']

    api = wandb.Api()

    runs = api.runs(path=f'{wandb.Api().default_entity}/{project_name}', filters={"group": group_name})

    output_dir = 'dl_wandb/downloaded_images'
    os.makedirs(output_dir, exist_ok=True)

    for run in runs:

        metrics = {}
        for col in ['accuracy', 'dice', 'iou', 't_mm', 't_s']:
            metrics[col] = run.summary.get(col)
        md_metrics = dict_to_markdown_table(metrics)
        with open('dl_wandb/metrics_%s.md' % str(run.name), 'w') as file:
            file.write(md_metrics)

        table = run.use_artifact(run.logged_artifacts()[1]).get(table_key)
        md_table = convert_to_markdown(table)
        with open('dl_wandb/table_%s.md' % str(run.name), 'w') as file:
            file.write(md_table)

        # images
        for media_key in media_keys:
            media_files = run.history(keys=[media_key]).get(media_key, [])
            for i, media in enumerate(media_files):
                if isinstance(media, dict) and 'path' in media:
                    image_url = 'https://api.wandb.ai/files/hahnec/' + project_name + '/' + run.url.split('runs/')[-1] + '/' + media['path']
                    base_name = '_'.join(os.path.basename(image_url).split('_')[:-1])+'.png' # skip hash clutter
                    image_name = f"{run.name}_{i}_{base_name}"
                    save_path = os.path.join(output_dir, image_name)
                    download_image(image_url, save_path)

    print("Done.")
