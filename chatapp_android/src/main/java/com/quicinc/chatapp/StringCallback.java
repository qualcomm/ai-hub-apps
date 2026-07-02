// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.chatapp;

/**
 * StringCallBack - Callback to tunnel JNI output into Java
 */
public interface StringCallback {
    void onNewString(String str);
}
