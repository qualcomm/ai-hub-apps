// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#include "Utilities.hpp"

#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

Utils::BoxCornerEncoding::BoxCornerEncoding(int a, int b, int c, int d, float sc, std::string name)
    : x1(a)
    , y1(b)
    , x2(c)
    , y2(d)
    , score(sc)
    , obj_label(name)
{
}

namespace Utils
{

std::vector<float> LoadImageFile(const std::string& image_path, uint32_t input_image_height, uint32_t input_image_width)
{
    cv::Mat image = cv::imread(image_path);

    if (image.empty())
    {
        std::ostringstream msg;
        msg << "Input image not found at " << image_path << ".\n Please provide full path to input image.";
        throw std::runtime_error(msg.str());
    }

    return PreprocessImage(image, input_image_height, input_image_width);
}

std::vector<float> PreprocessImage(const cv::Mat& input_image, uint32_t input_image_height, uint32_t input_image_width)
{
    cv::Mat image;
    cvtColor(input_image, image, cv::COLOR_BGR2RGB);
    resize(image, image, cv::Size(input_image_height, input_image_width));
    image = image.reshape(1, 1);

    std::vector<float> vec;
    image.convertTo(vec, CV_32FC1, 1. / 255);

    std::vector<float> output;
    output.reserve(vec.size());

    for (size_t ch = 0; ch < 3; ++ch)
    {
        for (size_t i = ch; i < vec.size(); i += 3)
        {
            output.push_back(vec[i]);
        }
    }
    return output;
}

float ComputeIntersectionOverUnion(const BoxCornerEncoding& box_i, const BoxCornerEncoding& box_j)
{
    const int box_i_y_min = std::min<int>(box_i.y1, box_i.y2);
    const int box_i_y_max = std::max<int>(box_i.y1, box_i.y2);
    const int box_i_x_min = std::min<int>(box_i.x1, box_i.x2);
    const int box_i_x_max = std::max<int>(box_i.x1, box_i.x2);
    const int box_j_y_min = std::min<int>(box_j.y1, box_j.y2);
    const int box_j_y_max = std::max<int>(box_j.y1, box_j.y2);
    const int box_j_x_min = std::min<int>(box_j.x1, box_j.x2);
    const int box_j_x_max = std::max<int>(box_j.x1, box_j.x2);

    const int area_i = (box_i_y_max - box_i_y_min) * (box_i_x_max - box_i_x_min);
    const int area_j = (box_j_y_max - box_j_y_min) * (box_j_x_max - box_j_x_min);

    if (area_i <= 0 || area_j <= 0)
    {
        return 0.0;
    }
    const int intersection_ymax = std::min<int>(box_i_y_max, box_j_y_max);
    const int intersection_xmax = std::min<int>(box_i_x_max, box_j_x_max);
    const int intersection_ymin = std::max<int>(box_i_y_min, box_j_y_min);
    const int intersection_xmin = std::max<int>(box_i_x_min, box_j_x_min);
    const int intersection_area = std::max<int>(intersection_ymax - intersection_ymin, 0) *
                                  std::max<int>(intersection_xmax - intersection_xmin, 0);
    return static_cast<float>(intersection_area) / static_cast<float>(area_i + area_j - intersection_area);
}

std::vector<BoxCornerEncoding> NonMaxSuppression(std::vector<BoxCornerEncoding> boxes, const float iou_threshold)
{

    if (boxes.size() == 0)
    {
        // No boxes to run NMS on, return empty vector
        return boxes;
    }

    std::sort(boxes.begin(), boxes.end(),
              [](const BoxCornerEncoding& left, const BoxCornerEncoding& right)
              {
                  if (left.score > right.score)
                  {
                      return true;
                  }
                  else
                  {
                      return false;
                  }
              });

    std::vector<bool> flag(boxes.size(), false);
    for (unsigned int i = 0; i < boxes.size(); i++)
    {
        if (flag[i])
        {
            continue;
        }

        for (unsigned int j = i + 1; j < boxes.size(); j++)
        {
            if (ComputeIntersectionOverUnion(boxes[i], boxes[j]) > iou_threshold)
            {
                flag[j] = true;
            }
        }
    }

    std::vector<BoxCornerEncoding> ret;
    for (unsigned int i = 0; i < boxes.size(); i++)
    {
        if (!flag[i])
        {
            ret.push_back(boxes[i]);
        }
    }

    return ret;
}

void AddBoundingBoxAndLabel(cv::Mat& image,
                            const Utils::BoxCornerEncoding& result,
                            const float ratio_h,
                            const float ratio_w)
{

    int y_top = static_cast<int>(result.y1 * ratio_h);
    int x_top = static_cast<int>(result.x1 * ratio_w);

    int y_bot = static_cast<int>(result.y2 * ratio_h);
    int x_bot = static_cast<int>(result.x2 * ratio_w);

    cv::Point p1(x_top, y_top);
    cv::Point p2(x_bot, y_bot);

    cv::rectangle(image, p1, p2, cv::Scalar(255, 0, 0), 5, cv::LINE_4);

    cv::Point label_position(p1.x, p1.y - 10);
    cv::Scalar label_color(255, 0, 0);
    int label_thickness = 4;
    int label_font = cv::FONT_HERSHEY_SIMPLEX;
    double label_scale = 2.5;
    cv::putText(image, result.obj_label, label_position, label_font, label_scale, label_color, label_thickness,
                cv::LINE_AA);
}

} // namespace Utils
