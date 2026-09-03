// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#pragma once

#include <string>

#include <opencv2/core.hpp>

namespace Utils
{

class BoxCornerEncoding
{

  public:
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    std::string obj_label;

    BoxCornerEncoding(int a, int b, int c, int d, float sc, std::string name = "default");
};

std::vector<float>
LoadImageFile(const std::string& image_path, uint32_t input_image_height, uint32_t input_image_width);

std::vector<float> PreprocessImage(const cv::Mat& input_image, uint32_t input_image_height, uint32_t input_image_width);

float ComputeIntersectionOverUnion(const BoxCornerEncoding& box_i, const BoxCornerEncoding& box_j);

std::vector<BoxCornerEncoding> NonMaxSuppression(std::vector<BoxCornerEncoding> boxes, const float iou_threshold);

void AddBoundingBoxAndLabel(cv::Mat& image, const BoxCornerEncoding& result, const float ratio_h, const float ratio_w);
} // namespace Utils
