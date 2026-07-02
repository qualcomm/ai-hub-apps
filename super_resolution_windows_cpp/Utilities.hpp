// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace Utils
{

void AddText(const cv::Mat& img, const std::string& msg);

std::vector<float>
LoadImageFile(const std::string& image_path, uint32_t input_image_height, uint32_t input_image_width);

} // namespace Utils
