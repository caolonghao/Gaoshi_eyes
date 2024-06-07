from flask import Flask, request, jsonify
import torch
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
import glob

app = Flask(__name__)

# Initialize model
model_folder = './gaoshi_model_pack'
use_folds = (0,)
device = torch.device('cuda', 2)

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
        # list all .png files in the input folder
        # debug: record running time
        # import time
        # start_time = time.time()
        # file_list = glob.glob(join(input_folder, '*.png'))
        # img_list, prop_list = [], []
        # for file in file_list:
        #     img, prop = NaturalImage2DIO().read_images([file])
        #     img_list.append(img)
        #     prop_list.append(prop)
        
        # load_time = time.time() - start_time
        
        # results = predictor.predict_from_list_of_npy_arrays(img_list,
        #                                             None,
        #                                             prop_list,
        #                                             None, 3, save_probabilities=False,
        #                                             num_processes_segmentation_export=2)
        
        results = predictor.predict_from_files(
            input_folder,
            None,
            save_probabilities=False, overwrite=True,
            num_processes_preprocessing=2, num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0
        )
        
        # computing_time = time.time() - start_time - load_time
        
        # print("Load time: ", load_time)
        # print("Computing time: ", computing_time)
        
        results = np.array(results)
        print(results)
        
        return jsonify({"status": "Prediction completed successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
