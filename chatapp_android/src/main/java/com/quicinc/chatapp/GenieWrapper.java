// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.chatapp;

/**
 * GenieWrapper: Class to connect JNI GenieWrapper and Java code
 */
public class GenieWrapper {
    long genieWrapperNativeHandle;

    /**
     * GenieWrapper: Loads model at provided path with provided htp config
     *
     * @param modelDirPath directory path on system pointing to model bundle
     * @param htpConfigPath HTP config file to use
     */
    GenieWrapper(String modelDirPath, String htpConfigPath) {
        genieWrapperNativeHandle = loadModel(modelDirPath, htpConfigPath);
    }

    /**
     * getResponseForPrompt: Generates response for provided user input
     *
     * @param userInput user input to generate response for
     * @param callback callback to tunnel each generated token to
     */
    public void getResponseForPrompt(String userInput, StringCallback callback) {
        getResponseForPrompt(genieWrapperNativeHandle, userInput, callback);
    }

    /**
     * finalize: Free previously loaded model
     */
    @Override
    protected void finalize() {
        freeModel(genieWrapperNativeHandle);
    }

    /**
     * loadModel: JNI method to load model using Genie C++ APIs
     *
     * @param modelDirPath directory path on system pointing to model bundle
     * @param htpConfigPath HTP config file to use
     * @return pointer to Genie C++ Wrapper to generate future responses
     */
    private native long loadModel(String modelDirPath, String htpConfigPath);

    /**
     * getResponseForPrompt: JNI method to generate response for provided user input
     *
     * @param nativeHandle native handle captured before with LoadModel
     * @param userInput user input to generate response for
     * @param callback callback to tunnel each generated token to
     */
    private native void getResponseForPrompt(long nativeHandle, String userInput, StringCallback callback);

    /**
     * FreeModel: JNI method to free previously loaded model
     *
     * @param nativeHandle native handle captured before with LoadModel
     */
    private native void freeModel(long nativeHandle);
}
