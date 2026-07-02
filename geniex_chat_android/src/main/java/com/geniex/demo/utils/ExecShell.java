// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo.utils;

import android.util.Log;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;

public class ExecShell {
    private static String LOG_TAG = ExecShell.class.getName();

    public enum SHELL_CMD {
        check_su_binary(new String[]{"/system/xbin/which", "su"});
        String[] command;

        SHELL_CMD(String[] command) {
            this.command = command;
        }
    }

    public ArrayList<String> executeCommand(String[] commands) {
        String line = null;
        ArrayList<String> fullResponse = new ArrayList<String>();
        Process localProcess = null;
        try {
            localProcess = Runtime.getRuntime().exec(commands);
        } catch (Exception e) {
            Log.e(LOG_TAG, "Command line execution failed", e);
            return null;
        }
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(localProcess.getOutputStream()));
        BufferedReader in = new BufferedReader(new InputStreamReader(localProcess.getInputStream()));
        try {
            while ((line = in.readLine()) != null) {
                fullResponse.add(line);
            }
        } catch (Exception e) {
            Log.e(LOG_TAG, "Command line result processing failed", e);
        }
        return fullResponse;
    }

    public ArrayList<String> executeCommand(SHELL_CMD shellCmd) {
        return executeCommand(shellCmd.command);
    }
}
