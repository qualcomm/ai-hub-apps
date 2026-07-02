// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.semanticsegmentation;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.view.WindowManager;
import android.widget.ProgressBar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.FragmentTransaction;

import com.quicinc.tflite.AIHubDefaults;

import java.io.IOException;
import java.security.NoSuchAlgorithmException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private ProgressBar progressBar;
    private SemanticSegmentation segmentor;
    ExecutorService backgroundTaskExecutor = Executors.newSingleThreadExecutor();
    Handler mainLooperHandler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        progressBar = (ProgressBar)findViewById(R.id.indeterminateBar);

        // Load model
        createTFLiteClassifiersAsync();
    }

    /**
     * Method to request Camera permission
     */
    private void cameraPermission() {
        requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
    }

    /**
     * Method to navigate to CameraFragment
     */
    private void overToCamera() {
        boolean passToFragment = MainActivity.this.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
        if (passToFragment) {
            if (segmentor != null) {
                FragmentTransaction transaction = getSupportFragmentManager().beginTransaction();
                transaction.add(R.id.main_content, CameraFragment.create(segmentor));
                transaction.commit();
            }
        } else {
            cameraPermission();
        }
    }

    void setLoadingUI(boolean loading) {
        runOnUiThread(() -> progressBar.setVisibility(loading ? View.VISIBLE : View.INVISIBLE));
    }

    /**
     * Create inference segmentor objects.
     * Loading the TF Lite model takes time, so this is done asynchronously to the main UI thread.
     * Disables the inference UI during load and reenables it afterwards.
     */
    void createTFLiteClassifiersAsync() {
        if (segmentor != null) {
            throw new RuntimeException("Segmentor was already created");
        }
        setLoadingUI(true);

        // Exit the UI thread and instantiate the model in the background.
        backgroundTaskExecutor.execute(() -> {
            // Create a SemanticSegmentation object with all available compute units (NPU, GPU, CPU)
            String tfLiteModelAsset = this.getResources().getString(R.string.tfLiteModelAsset);
            try {
                segmentor = new SemanticSegmentation(
                        this,
                        tfLiteModelAsset,
                        AIHubDefaults.delegatePriorityOrder /* AI Hub Defaults */
                );
            } catch (IOException | NoSuchAlgorithmException e) {
                throw new RuntimeException(e.getMessage());
            }
            setLoadingUI(false);
            mainLooperHandler.post(this::overToCamera);
        });
    }


    @Override
    protected void onResume()
    {
        super.onResume();
        overToCamera();
    }
    @Override
    protected void onStop() {
        super.onStop();
    }
}
