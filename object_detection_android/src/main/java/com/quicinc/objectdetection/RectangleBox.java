// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.objectdetection;

import java.util.ArrayList;

/**
 * RectangleBox class defines the property associated with each box like coordinates labels,
 * confidence etc. Can also create copy of boxes.
 */
public class RectangleBox {

    public float top;
    public float bottom;
    public float left;
    public float right;

    public int classIdx;
    public String label;
    public float confidence;

    public static ArrayList<RectangleBox> createBoxes(int num) {
        final ArrayList<RectangleBox> boxes;
        boxes = new ArrayList<>();
        for (int i = 0; i < num; ++i) {
            boxes.add(new RectangleBox());
        }
        return boxes;
    }
}
