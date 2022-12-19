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
from ikomia.utils import pyqtutils, qtconversion
from infer_huggingface_semantic_segmentation.infer_huggingface_semantic_segmentation_process import InferHuggingFaceSemanticSegmentationParam
from torch.cuda import is_available
# PyQt GUI framework
from PyQt5.QtWidgets import *
from infer_huggingface_semantic_segmentation.utils import Autocomplete
import os
import timm

# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferHuggingFaceSemanticSegmentationWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferHuggingFaceSemanticSegmentationParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

 

        # Loading model from list
        model_list_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        "model_list.txt")
        model_list_file = open(model_list_path, "r")

        model_list = model_list_file.read()
        model_list = model_list.split("\n")
        self.combo_model = Autocomplete(model_list, parent=None, i=True, allow_duplicates=False)
        self.label_model = QLabel("Model name")
        self.gridLayout.addWidget(self.combo_model, 0, 1)
        self.gridLayout.addWidget(self.label_model, 0, 0)
        self.combo_model.setCurrentText(self.parameters.model_name)
        model_list_file.close()

        self.check_checkoint = pyqtutils.append_check(self.gridLayout, "Model from checkpoint(local)",
                                                       self.parameters.checkpoint)

        self.check_checkoint.stateChanged.connect(self.onStateChanged)

        self.combo_model.setVisible(not self.check_checkoint.isChecked())
        
        # Loading moadel from checkpoint path
        self.browse_ckpt = pyqtutils.append_browse_file(self.gridLayout,
                                                        label="Checkpoint path",
                                                        path=self.parameters.checkpoint_path,
                                                        mode=QFileDialog.Directory)

        self.browse_ckpt.setVisible(self.check_checkoint.isChecked())

        # Cuda
        self.check_cuda = pyqtutils.append_check(
                        self.gridLayout, "Cuda",
                        self.parameters.cuda and is_available())

        # Background label
        self.check_background = pyqtutils.append_check(
                                                self.gridLayout, "Background label set at index 0",
                                                self.parameters.background
                                                )

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.setLayout(layout_ptr)

    # Widget update on check
    def onStateChanged(self, int):
        self.browse_ckpt.setVisible(self.check_checkoint.isChecked())
        self.combo_model.setVisible(not self.check_checkoint.isChecked())

    def onApply(self):
        # Apply button clicked slot
        self.parameters.update = True
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.checkpoint = self.check_checkoint.isChecked()
        self.parameters.checkpoint_path = self.browse_ckpt.path
        self.parameters.background = self.check_background.isChecked()
        # Send signal to launch the process
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferHuggingFaceSemanticSegmentationWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_huggingface_semantic_segmentation"

    def create(self, param):
        # Create widget object
        return InferHuggingFaceSemanticSegmentationWidget(param, None)
