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
import numpy as np
import random
import os
from torch import nn
import json 


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferHuggingFaceSemanticSegmentationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.cuda = torch.cuda.is_available()
        self.model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
        self.checkpoint = False
        self.checkpoint_path = ""
        self.background = False
        self.update = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = strtobool(param_map["cuda"])
        self.model_name = param_map["model_name"]
        self.pretrained = strtobool(param_map["checkpoint"])
        self.checkpoint_path = param_map["checkpoint_path"]
        self.background = strtobool(param_map["background_idx"])
        self.update = strtobool(param_map["update"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["cuda"] = str(self.cuda)
        param_map["model_name"] = str(self.model_name)
        param_map["checkpoint"] = str(self.checkpoint)
        param_map["checkpoint_path"] = self.checkpoint_path
        param_map["background_idx"] = str(self.background)
        param_map["update"] = str(self.update)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferHuggingFaceSemanticSegmentation(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        self.addOutput(dataprocess.CSemanticSegIO())
        if param is None:
            self.setParam(InferHuggingFaceSemanticSegmentationParam())
        else:
            self.setParam(copy.deepcopy(param))

        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_extractor = None
        self.colors = None
        self.classes = None
        self.update = False

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def infer(self, image):
        param = self.getParam()

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

        # Get output :
        semantic_output = self.getOutput(1)

        # dstImage
        dst_image = pred_seg.cpu().detach().numpy().astype(dtype=np.uint8)

        # Get mask
        self.setOutputColorMap(0, 1, self.colors)

        semantic_output.setMask(dst_image)

        semantic_output.setClassNames(self.classes, self.colors)

        self.forwardInputImage(0, 0)

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        image_in = self.getInput(0)

        # Get image from input/output (numpy array):
        image = image_in.getImage()

        param = self.getParam()

        if param.update or self.model is None:
            model_id = None
            # Feature extractor selection
            if param.checkpoint is False:
                model_id = param.model_name
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            else:
                model_list_path = os.path.join(param.checkpoint_path, "config.json")
                model_id = param.checkpoint_path
                with open(str(model_list_path), "r") as f:
                    model_name_list = json.load(f)
                feature_extractor_id = model_name_list['_name_or_path']
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_id)

            # Loading model weight
            self.model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")
            self.model.to(self.device)
            print("Will run on {}".format(self.device.type))

            # Get label name
            self.classes = list(self.model.config.id2label.values())

            # Color palette
            n = len(self.classes)
            random.seed(14)
            if param.background is True:
                self.colors = [[0,0,0]]
                for i in range(n-1):
                    self.colors.append(random.choices(range(256), k=3))
            else:
                self.colors = []
                for i in range(n):
                    self.colors.append(random.choices(range(256), k=3))

            param.update = False

        # Inference
        self.infer(image)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferHuggingFaceSemanticSegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_huggingface_semantic_segmentation"
        self.info.shortDescription = "Semantic segmentation using models from Hugging Face."
        self.info.description = "This plugin proposes inference for semantic segmentation"\
                                "using transformers models from Hugging Face. It regroups"\
                                "models covered by the Hugging Face class: "\
                                "AutoModelForSemanticSegmentation. Models can be loaded either"\
                                "from your fine-tuned model (local) or from the Hugging Face Hub."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/icon.png"
        self.info.authors = "Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond,"\
                            "Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault,"\
                            "Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer,"\
                            "Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu,"\
                            "Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame,"\
                            "Quentin Lhoest, Alexander M. Rush"
        self.info.article = "Huggingface's Transformers: State-of-the-art Natural Language Processing"
        self.info.journal = "EMNLP"
        self.info.license = "Apache License Version 2.0"
        # URL of documentation
        self.info.documentationLink = "https://www.aclweb.org/anthology/2020.emnlp-demos.6"
        # Code source repository
        self.info.repository = "https://github.com/huggingface/transformers"
        # Keywords used for search
        self.info.keywords = "semantic, segmentation, inference, transformer,"\
                            "Hugging Face, Pytorch, Segformer, DPT, Beit, data2vec"

    def create(self, param=None):
        # Create process object
        return InferHuggingFaceSemanticSegmentation(self.info.name, param)
