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

        # Cuda
        self.check_cuda = pyqtutils.append_check(
                        self.gridLayout, "Cuda",
                        self.parameters.cuda and is_available())

        # Model loading method
        self.combo_model = pyqtutils.append_combo(self.gridLayout, "Model loading")
        self.combo_model.addItem("From Hugging Face Model Hub")
        self.combo_model.addItem("From checkpoint (local)")
        self.combo_model.setCurrentText(self.parameters.model_loading)

        # Model
        self.load_model = pyqtutils.append_edit(
                                                self.gridLayout,
                                                "Model name/Checkpoint path",
                                                self.parameters.model_card
                                                )

        # Background label
        self.check_background = pyqtutils.append_check(
                                                self.gridLayout, "Background label set to index 0",
                                                self.parameters.background
                                                )

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.setLayout(layout_ptr)

    def onApply(self):
        # Apply button clicked slot
        self.parameters.update = True
        self.parameters.model_loading = self.combo_model.currentText()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.model_card = self.load_model.text()
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
