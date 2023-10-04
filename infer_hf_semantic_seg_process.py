# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
import copy
# Your imports below
import transformers
from transformers import AutoFeatureExtractor, AutoModelForSemanticSegmentation
from ikomia.utils import strtobool
import numpy as np
import torch
import os
from torch import nn
import json 


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferHfSemanticSegParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.cuda = torch.cuda.is_available()
        self.model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
        self.use_custom_model = False
        self.model_weight_file = ""
        self.update = False

    def set_values(self, params):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = strtobool(params["cuda"])
        self.model_name = params["model_name"]
        self.pretrained = strtobool(params["use_custom_model"])
        self.model_weight_file = params["model_weight_file"]
        self.update = strtobool(params["update"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        params = {
            "cuda": str(self.cuda),
            "model_name": str(self.model_name),
            "use_custom_model": str(self.use_custom_model),
            "model_weight_file": self.model_weight_file,
            "update": str(self.update)}
        return params


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferHfSemanticSeg(dataprocess.CSemanticSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CSemanticSegmentationTask.__init__(self, name)
        if param is None:
            self.set_param_object(InferHfSemanticSegParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_extractor = None
        self.classes = None
        self.update = False
        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")


    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def infer(self, image):
        param = self.get_param_object()

        # Image pre-pocessing (image transformation and conversion to PyTorch tensor)
        pixel_val = self.feature_extractor(image, return_tensors="pt", resample=0).pixel_values
        if param.cuda is True:
            pixel_val = pixel_val.cuda()

        # Prediction
        outputs = self.model(pixel_val)
        logits = outputs.logits

        upsampled_logits = nn.functional.interpolate(
            logits,
            size = image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]

        # dstImage
        dst_image = pred_seg.cpu().detach().numpy().astype(dtype=np.uint8)

        # Get mask
        self.set_names(self.classes)

        self.set_mask(dst_image)

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        image_in = self.get_input(0)

        # Get image from input/output (numpy array):
        image = image_in.get_image()

        param = self.get_param_object()

        if param.update or self.model is None:
            param = self.get_param_object()
            model_id = None
            # Feature extractor selection
            if param.model_weight_file == "": # Check if the model is from local or from huggingface hub
                model_id = param.model_name
                try:
                    self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                        model_id,
                        cache_dir=self.model_folder,
                        local_files_only=True
                    )
                except Exception as e:
                    print(f"Failed with error: {e}. Trying without the local_files_only parameter...")
                    self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                        model_id,
                        cache_dir=self.model_folder
                    )    

            else:
                model_folder_path = os.path.dirname(param.model_weight_file)
                model_list_path = os.path.join(model_folder_path, "config.json")
                model_id = model_folder_path
                with open(str(model_list_path), "r") as f:
                    model_name_list = json.load(f)
                feature_extractor_id = model_name_list['_name_or_path']
                try: 
                    self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                        feature_extractor_id,
                        cache_dir=self.model_folder,
                        local_files_only=True
                    )
                except Exception as e:
                    print(f"Failed with error: {e}. Trying without the local_files_only parameter...")
                    self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                        feature_extractor_id,
                        cache_dir=self.model_folder,
                    )

            # Loading model weight
            try: 
                self.model = AutoModelForSemanticSegmentation.from_pretrained(
                    model_id,
                    cache_dir=self.model_folder,
                    local_files_only=True
                    )
            except Exception as e:
                print(f"Failed with error: {e}. Trying without the local_files_only parameter...") 
                self.model = AutoModelForSemanticSegmentation.from_pretrained(
                    model_id,
                    cache_dir=self.model_folder,
                    )
        
            self.device = torch.device("cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
            self.model.to(self.device)

            print("Will run on {}".format(self.device.type))

            # Get label name
            self.classes = list(self.model.config.id2label.values())

            param.update = False

        # Inference
        self.infer(image)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferHfSemanticSegFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_hf_semantic_seg"
        self.info.short_description = "Semantic segmentation using models from Hugging Face."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.1.1"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, "\
                            "Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, "\
                            "Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, "\
                            "Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, "\
                            "Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, "\
                            "Quentin Lhoest, Alexander M. Rush"
        self.info.article = "Huggingface's Transformers: State-of-the-art Natural Language Processing"
        self.info.journal = "EMNLP"
        self.info.license = "Apache License Version 2.0"
        # URL of documentation
        self.info.documentation_link = "https://www.aclweb.org/anthology/2020.emnlp-demos.6"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_hf_semantic_seg"
        self.info.original_repository = "https://github.com/huggingface/transformers"
        # Keywords used for search
        self.info.keywords = "semantic, segmentation, inference, transformer,"\
                            "Hugging Face, Pytorch, Segformer, DPT, Beit, data2vec"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "SEMANTIC_SEGMENTATION"

    def create(self, param=None):
        # Create process object
        return InferHfSemanticSeg(self.info.name, param)
