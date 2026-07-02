// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.superresolution;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import com.quicinc.ImageProcessing;
import com.quicinc.tflite.AIHubDefaults;

import java.io.IOException;
import java.io.InputStream;
import java.security.NoSuchAlgorithmException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.HashSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class MainActivity extends AppCompatActivity {
    // UI Elements
    RadioGroup delegateSelectionGroup;
    RadioButton allDelegatesButton;
    RadioButton cpuOnlyButton;
    ImageView selectedImageView;
    TextView inferenceTimeView;
    TextView predictionTimeView;
    Spinner imageSelector;
    Button predictionButton;
    ActivityResultLauncher<Intent> selectImageResultLauncher;
    private final String fromGalleryImageSelectorOption = "From Gallery";
    private final String notSelectedImageSelectorOption = "Not Selected";
    private final String[] imageSelectorOptions =
            { notSelectedImageSelectorOption,
                    "Sample1.jpg",
                    "Sample2.jpg",
                    fromGalleryImageSelectorOption};

    // Inference Elements
    Bitmap selectedImage = null; // Raw image, not resized
    private SuperResolution defaultDelegateUpscaler;
    private SuperResolution cpuOnlyUpscaler;
    private boolean cpuOnlyClassification = false;
    NumberFormat timeFormatter = new DecimalFormat("0.00");
    ExecutorService backgroundTaskExecutor = Executors.newSingleThreadExecutor();
    Handler mainLooperHandler = new Handler(Looper.getMainLooper());

    /**
     * Instantiate the activity on first load.
     * Creates the UI and a background thread that instantiates the upscaler  TFLite model.
     *
     * @param savedInstanceState Saved instance state.
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //
        // UI Initialization
        //
        setContentView(R.layout.main_activity);
        selectedImageView = (ImageView) findViewById(R.id.selectedImageView);
        delegateSelectionGroup = (RadioGroup) findViewById(R.id.delegateSelectionGroup);
        cpuOnlyButton = (RadioButton)findViewById(R.id.cpuOnlyRadio);
        allDelegatesButton = (RadioButton)findViewById(R.id.defaultDelegateRadio);

        imageSelector = (Spinner) findViewById((R.id.imageSelector));
        inferenceTimeView = (TextView)findViewById(R.id.inferenceTimeResultText);
        predictionTimeView = (TextView)findViewById(R.id.predictionTimeResultText);
        predictionButton = (Button)findViewById(R.id.runModelButton);

        // Setup Image Selector Dropdown
        ArrayAdapter ad = new ArrayAdapter(this, android.R.layout.simple_spinner_item, imageSelectorOptions);
        ad.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        imageSelector.setAdapter(ad);
        imageSelector.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                // Load selected picture from assets
                ((TextView) view).setTextColor(getResources().getColor(R.color.white));
                if (!parent.getItemAtPosition(position).equals(notSelectedImageSelectorOption)) {
                    if (parent.getItemAtPosition(position).equals(fromGalleryImageSelectorOption)) {
                        Intent i = new Intent();
                        i.setType("image/*");
                        i.setAction(Intent.ACTION_GET_CONTENT);
                        selectImageResultLauncher.launch(i);
                    } else {
                        loadImageFromStringAsync((String) parent.getItemAtPosition(position));
                    }
                } else {
                    displayDefaultImage();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) { }
        });

        // Setup Image Selection from Phone Gallery
        selectImageResultLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                (ActivityResult result) -> {
                    if (result.getResultCode() == Activity.RESULT_OK &&
                            result.getData() != null &&
                            result.getData().getData() != null) {
                        loadImageFromURIAsync((Uri)(result.getData().getData()));
                    } else {
                        displayDefaultImage();
                    }
                });

        // Setup delegate selection buttons
        delegateSelectionGroup.setOnCheckedChangeListener((group, checkedId) -> {
            if (checkedId == R.id.cpuOnlyRadio) {
                if (!cpuOnlyClassification) {
                    this.cpuOnlyClassification = true;
                    clearPredictionResults();
                }
            } else if (checkedId == R.id.defaultDelegateRadio) {
                if (cpuOnlyClassification) {
                    this.cpuOnlyClassification = false;
                    clearPredictionResults();
                }
            } else {
                throw new RuntimeException("A radio button for selected runtime is not implemented");
            }
        });

        // Setup button callback
        predictionButton.setOnClickListener((view) -> updatePredictionDataAsync());

        // Exit the UI thread and instantiate the model in the background.
        createTFLiteUpscalerAsync();

        // Enable image selection
        enableImageSelector();
        enableDelegateSelectionButtons();
    }

    /**
     * Enable or disable UI controls for inference.
     *
     * @param enabled If true, enable the UI. If false, disable the UI.
     */
    void setInferenceUIEnabled(boolean enabled) {
        if (!enabled) {
            inferenceTimeView.setText("-- ms");
            predictionTimeView.setText("-- ms");
            predictionButton.setEnabled(false);
            predictionButton.setAlpha(0.5f);
            imageSelector.setEnabled(false);
            imageSelector.setAlpha(0.5f);
            cpuOnlyButton.setEnabled(false);
            allDelegatesButton.setEnabled(false);
        } else if (cpuOnlyUpscaler != null && defaultDelegateUpscaler != null && selectedImage != null) {
            predictionButton.setEnabled(true);
            predictionButton.setAlpha(1.0f);
            enableImageSelector();
            enableDelegateSelectionButtons();
        }
    }

    /**
     * Enable the image selector UI spinner.
     */
    void enableImageSelector() {
        imageSelector.setEnabled(true);
        imageSelector.setAlpha(1.0f);
    }

    /**
     * Enable the image selector UI radio buttons.
     */
    void enableDelegateSelectionButtons() {
        cpuOnlyButton.setEnabled(true);
        allDelegatesButton.setEnabled(true);
    }

    /**
     * Reset the selected image view to the default image,
     * and enable portions of the inference UI accordingly.
     */
    void displayDefaultImage() {
        setInferenceUIEnabled(false);
        enableImageSelector();
        enableDelegateSelectionButtons();
        clearPredictionResults();
        selectedImageView.setImageResource(R.drawable.ic_launcher_background);
        selectedImage = null;
    }

    /**
     * Clear previous inference results from the UI.
     */
    void clearPredictionResults() {
        if (selectedImage != null) {
            selectedImageView.setImageBitmap(selectedImage);
        }
        inferenceTimeView.setText("-- ms");
        predictionTimeView.setText("-- ms");
    }

    /**
     * Load an image for inference and update the UI accordingly.
     * The image will be loaded asynchronously to the main UI thread.
     *
     * @param imagePath Path to the image relative to the the `assets/images/` folder
     */
    void loadImageFromStringAsync(String imagePath) {
        setInferenceUIEnabled(false);
        // Exit the main UI thread and load the image in the background.
        backgroundTaskExecutor.execute(() -> {
            // Background task
            try (InputStream inputImage = getAssets().open("images/" + imagePath)) {
                selectedImage = BitmapFactory.decodeStream(inputImage);
                // Downscale the image to the size the model supports.
                int[] inputSize = defaultDelegateUpscaler.getInputWidthHeight();
                selectedImage = ImageProcessing.resizeAndPadMaintainAspectRatio(selectedImage, inputSize[0], inputSize[1], 0xFF);
            } catch (IOException e) {
                throw new RuntimeException(e.getMessage());
            }

            mainLooperHandler.post(() -> {
                // In main UI thread
                selectedImageView.setImageBitmap(selectedImage);
                setInferenceUIEnabled(true);
            });
        });
    }

    /**
     * Load an image for inference and update the UI accordingly.
     * The image will be loaded asynchronously to the main UI thread.
     *
     * @param imageUri URI to the image.
     */
    void loadImageFromURIAsync(Uri imageUri) {
        setInferenceUIEnabled(false);
        // Exit the main UI thread and load the image in the background.
        backgroundTaskExecutor.execute(() -> {
            // Background task
            try {
                if (android.os.Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    selectedImage = ImageDecoder.decodeBitmap(ImageDecoder.createSource(getContentResolver(), imageUri), (decoder, info, src) -> {
                        decoder.setMutableRequired(true);
                    });
                } else {
                    selectedImage = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                }
                // Downscale the image to the size the model supports.
                int[] inputSize = defaultDelegateUpscaler.getInputWidthHeight();
                selectedImage = ImageProcessing.resizeAndPadMaintainAspectRatio(selectedImage, inputSize[0], inputSize[1], 0xFF);
            } catch (IOException e) {
                throw new RuntimeException(e.getMessage());
            }

            mainLooperHandler.post(() -> {
                // In main UI thread
                selectedImageView.setImageBitmap(selectedImage);
                setInferenceUIEnabled(true);
            });
        });
    }

    /**
     * Run the upscaler on the currently selected image.
     * Prediction will run asynchronously to the main UI thread.
     * Disables inference UI before inference and re-enables it afterwards.
     */
    void updatePredictionDataAsync() {
        setInferenceUIEnabled(false);

        SuperResolution imageClassification;
        if (cpuOnlyClassification) {
            imageClassification = cpuOnlyUpscaler;
        } else {
            imageClassification = defaultDelegateUpscaler;
        }

        // Exit the main UI thread and execute the model in the background.
        backgroundTaskExecutor.execute(() -> {
            // Background task
            Bitmap result = imageClassification.generateUpscaledImage(selectedImage);
            long inferenceTime = imageClassification.getLastInferenceTime();
            long predictionTime = imageClassification.getLastPostprocessingTime() + inferenceTime + imageClassification.getLastPreprocessingTime();
            String inferenceTimeText = timeFormatter.format((double) inferenceTime / 1000000);
            String predictionTimeText = timeFormatter.format((double) predictionTime / 1000000);

            mainLooperHandler.post(() -> {
                // In main UI thread
                selectedImageView.setImageBitmap(result);
                inferenceTimeView.setText(inferenceTimeText + " ms");
                predictionTimeView.setText(predictionTimeText + " ms");
                setInferenceUIEnabled(true);
            });
        });
    }

    /**
     * Create inference upscaler objects.
     * Loading the TF Lite model takes time, so this is done asynchronously to the main UI thread.
     * Disables the inference UI during load and reenables it afterwards.
     */
    void createTFLiteUpscalerAsync() {
        if (defaultDelegateUpscaler != null || cpuOnlyUpscaler != null) {
            throw new RuntimeException("Classifiers were already created");
        }
        setInferenceUIEnabled(false);

        // Exit the UI thread and instantiate the model in the background.
        backgroundTaskExecutor.execute(() -> {
            // Create two upscalers.
            // One uses the default set of delegates (can access NPU, GPU, CPU), and the other uses only XNNPack (CPU).
            String tfLiteModelAsset = this.getResources().getString(R.string.tfLiteModelAsset);
            try {
                defaultDelegateUpscaler = new SuperResolution(
                        this,
                        tfLiteModelAsset,
                        AIHubDefaults.delegatePriorityOrder /* AI Hub Defaults */
                );
                cpuOnlyUpscaler = new SuperResolution(
                        this,
                        tfLiteModelAsset,
                        AIHubDefaults.delegatePriorityOrderForDelegates(new HashSet<>() /* No delegates; cpu only */)
                );
            } catch (IOException | NoSuchAlgorithmException e) {
                throw new RuntimeException(e.getMessage());
            }

            mainLooperHandler.post(() -> setInferenceUIEnabled(true));
        });
    }

    /**
     * Destroy this activity and release memory used by held objects.
     */
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cpuOnlyUpscaler != null) cpuOnlyUpscaler.close();
        if (defaultDelegateUpscaler != null) defaultDelegateUpscaler.close();
    }
}
