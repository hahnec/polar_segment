from pathlib import Path

base_path = '/home/chris/Datasets/03_HORAO/TumorMeasurementsCalib/'
fn = 'histology/labels_augmented_GM_WM_masked_FG_no_border.png'

multilabel_tumor_samples = []
for b in range(1, 6):
    batch_path = Path(base_path) / ('batch' + str(b))
    for seq_path in batch_path.iterdir():
        for set_path in seq_path.iterdir(): 
            if (set_path / fn).exists():
                trunc_path = str(set_path.relative_to(base_path)) + '/'
                multilabel_tumor_samples.append(trunc_path)

from natsort import natsorted
multilabel_tumor_samples = natsorted(multilabel_tumor_samples)
output_file = Path(base_path) / 'cases' / 'multilabel_tumor_samples.txt'
if output_file.exists(): output_file.unlink()
with open(output_file, "a") as f:
    for el in multilabel_tumor_samples:
        f.write(el)
        f.write('\n')
