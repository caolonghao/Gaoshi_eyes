from flask import Flask, request, jsonify
import torch
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

app = Flask(__name__)

# Initialize model
model_folder = './gaoshi_model_pack'
use_folds = (0,)
device = torch.device('cuda', 0)

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=device,
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
)

predictor.initialize_from_trained_model_folder(
    model_folder,
    use_folds=use_folds,
    checkpoint_name='checkpoint_best.pth',
)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_folder = data.get('input_folder')
    output_folder = data.get('output_folder')

    if not input_folder or not output_folder:
        return jsonify({"error": "Invalid input or output folder"}), 400

    try:
        predictor.predict_from_files(
            input_folder,
            output_folder,
            save_probabilities=False, overwrite=True,
            num_processes_preprocessing=2, num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0
        )
        
        results = predictor.predict_from_files(
            input_folder,
            None,
            save_probabilities=False, overwrite=True,
            num_processes_preprocessing=2, num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0
        )
        return jsonify({"status": "Prediction completed successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
