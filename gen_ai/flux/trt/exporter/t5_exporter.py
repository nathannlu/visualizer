#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from ...modules.conditioner import HFEmbedder
from .base_exporter import BaseExporter, TransformersModelWrapper
from ..mixin import T5Mixin


class T5Exporter(T5Mixin, BaseExporter):
    def __init__(
        self,
        model: HFEmbedder,
        fp16=False,
        tf32=True,
        bf16=False,
        max_batch=8,
        build_strongly_typed=False,
        verbose=True,
    ):
        exp_model = TransformersModelWrapper(model=model, output_name="last_hidden_state")
        super().__init__(
            text_maxlen=model.max_length,
            hidden_size=model.hf_module.config.hidden_size,
            model=exp_model,
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            max_batch=max_batch,
            verbose=verbose,
        )
        self.build_strongly_typed = build_strongly_typed
        # set proper dtype
        self.prepare_model()

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        return ["text_embeddings"]

    def get_dynamic_axes(self):
        dynamic_axes = {
            "input_ids": {0: "B"},
            "text_embeddings": {0: "B"},
        }
        return dynamic_axes

    def check_dims(
        self,
        batch_size: int,
    ) -> None:
        assert batch_size >= self.min_batch and batch_size <= self.max_batch

    def get_sample_input(
        self,
        batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
    ) -> torch.Tensor:
        self.check_dims(batch_size)
        return torch.zeros(
            (batch_size, self.text_maxlen),
            dtype=torch.int32,
            device=self.device,
        )

    def get_input_profile(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
        static_batch: bool,
        static_shape: bool,
    ):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch

        self.check_dims(batch_size)
        return {
            "input_ids": [
                (min_batch, self.text_maxlen),
                (batch_size, self.text_maxlen),
                (max_batch, self.text_maxlen),
            ]
        }
