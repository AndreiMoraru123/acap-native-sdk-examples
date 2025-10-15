/**
 * Copyright (C) 2025, Axis Communications AB, Lund, Sweden
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * - licence_plate_yolov8.c -
 *
 * This application loads a larod YOLOv5 model which takes an image as input. The output is
 * YOLOv5-specifically parsed to retrieve values corresponding to the class, score and location of
 * detected objects in the image.
 *
 * The application expects two arguments on the command line in the
 * following order: MODELFILE LABELSFILE.
 *
 * First argument, MODELFILE, is a string describing path to the model.
 *
 * Second argument, LABELSFILE, is a string describing path to the label txt.
 *
 */

#include "argparse.h"
#include "imgprovider.h"
#include "labelparse.h"
#include "model.h"
#include "model_params.h"  //Generated at build time
#include "panic.h"
#include "vdo-error.h"
#include "vdo-frame.h"
#include "vdo-types.h"
#include <axsdk/axparameter.h>

#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <syslog.h>

#define APP_NAME "object_detection_yolov5"

volatile sig_atomic_t running = 1;

static void shutdown(int status) {
    (void)status;
    running = 0;
}

typedef struct model_params {
    int input_width;
    int input_height;
    float quantization_scale;
    float quantization_zero_point;
    int num_classes;
    int num_detections;
    int size_per_detection;
} model_params_t;

static void parse_licence_plate(uint8_t* tensor,
                                model_params_t* model_params,
                                char** labels,
                                size_t num_labels,
                                char* plate_string,
                                size_t max_string_len) {
    int sequence_length = model_params->num_detections;
    int num_classes     = model_params->num_classes;
    float qt_zero_point = model_params->quantization_zero_point;
    float qt_scale      = model_params->quantization_scale;

    int string_idx  = 0;
    plate_string[0] = '\0';

    for (int pos = 0; pos < sequence_length; pos++) {
        if (string_idx >= (int)max_string_len - 1) {
            break;
        }

        float max_prob     = -1.0f;
        int best_class_idx = -1;
        for (int cls = 0; cls < num_classes; cls++) {
            int tensor_idx = pos * num_classes + cls;  // flattened tensor index

            float probability = (tensor[tensor_idx] - qt_zero_point) * qt_scale;  // de-quantize

            if (probability > max_prob) {
                max_prob       = probability;
                best_class_idx = cls;
            }
        }

        if (best_class_idx > 0 && best_class_idx < (int)num_labels && max_prob > 0.0) {
            char* predicted_char = labels[best_class_idx];
            if (predicted_char && strlen(predicted_char) > 0) {
                if (string_idx == 0 || plate_string[string_idx - 1] != predicted_char[0]) {
                    plate_string[string_idx++] = predicted_char[0];
                    plate_string[string_idx]   = '\0';
                }
            }
        }
    }
    syslog(LOG_INFO, "Detected license plate: %s", plate_string);
}


static unsigned int elapsed_ms(struct timeval* start_ts, struct timeval* end_ts) {
    return (unsigned int)(((end_ts->tv_sec - start_ts->tv_sec) * 1000) +
                          ((end_ts->tv_usec - start_ts->tv_usec) / 1000));
}

int main(int argc, char** argv) {
    g_autoptr(GError) vdo_error           = NULL;
    img_provider_t* image_provider        = NULL;
    model_provider_t* model_provider      = NULL;
    model_tensor_output_t* tensor_outputs = NULL;

    // Stop main loop at signal
    signal(SIGTERM, shutdown);
    signal(SIGINT, shutdown);

    args_t args;
    parse_args(argc, argv, &args);

    model_params_t* model_params = (model_params_t*)malloc(sizeof(model_params_t));
    if (model_params == NULL) {
        panic("%s: Unable to allocate model_params_t: %s", __func__, strerror(errno));
    }

    // Comes from model_params.h
    model_params->input_width             = MODEL_INPUT_WIDTH;
    model_params->input_height            = MODEL_INPUT_HEIGHT;
    model_params->quantization_scale      = QUANTIZATION_SCALE;
    model_params->quantization_zero_point = QUANTIZATION_ZERO_POINT;
    model_params->num_classes             = NUM_CLASSES;
    model_params->num_detections          = NUM_DETECTIONS;
    model_params->size_per_detection =
        5 + NUM_CLASSES;  // Each detection consists of [x, y, w, h, object_likelihood,
                          // class1_likelihood, class2_likelihood, class3_likelihood, ... ]

    syslog(LOG_INFO,
           "Model input size w/h: %d x %d",
           model_params->input_width,
           model_params->input_height);
    syslog(LOG_INFO, "Quantization scale: %f", model_params->quantization_scale);
    syslog(LOG_INFO, "Quantization zero point: %f", model_params->quantization_zero_point);
    syslog(LOG_INFO, "Number of classes: %d", model_params->num_classes);
    syslog(LOG_INFO, "Number of detections: %d", model_params->num_detections);

    // Create a new axparameter instance
    GError* axparameter_error       = NULL;
    AXParameter* axparameter_handle = ax_parameter_new(APP_NAME, &axparameter_error);
    if (axparameter_handle == NULL) {
        panic("%s", axparameter_error->message);
    }

    ax_parameter_free(axparameter_handle);

    VdoFormat vdo_format = VDO_FORMAT_YUV;
    double vdo_framerate = 30.0;

    if (!g_strcmp0(args.device_name, "a9-dlpu-tflite")) {
        // Possible to run RGB on ARTPEC-9
        vdo_format = VDO_FORMAT_RGB;
    }

    // Choose a valid stream resolution since only certain resolutions are allowed
    unsigned int stream_width  = 0;
    unsigned int stream_height = 0;
    if (!choose_stream_resolution(model_params->input_width,
                                  model_params->input_height,
                                  vdo_format,
                                  "native",
                                  "all",
                                  &stream_width,
                                  &stream_height)) {
        syslog(LOG_ERR, "%s: Failed choosing stream resolution", __func__);
        goto end;
    }
    syslog(LOG_INFO,
           "Creating VDO image provider and creating stream %u x %u",
           stream_width,
           stream_height);

    image_provider = create_img_provider(stream_width, stream_height, 2, vdo_format, vdo_framerate);
    if (!image_provider) {
        panic("%s: Could not create image provider", __func__);
    }

    size_t number_output_tensors = 0;
    model_provider               = create_model_provider(model_params->input_width,
                                           model_params->input_height,
                                           image_provider->width,
                                           image_provider->height,
                                           image_provider->pitch,
                                           image_provider->format,
                                           VDO_FORMAT_RGB,
                                           args.model_file,
                                           args.device_name,
                                           false,
                                           &number_output_tensors);
    if (!model_provider) {
        panic("%s: Could not create model provider", __func__);
    }
    tensor_outputs = calloc(number_output_tensors, sizeof(model_tensor_output_t));
    if (!tensor_outputs) {
        panic("%s: Could not allocate tensor outputs", __func__);
    }

    char** labels = NULL;          // This is the array of label strings. The label
                                   // entries points into the large label_file_data buffer.
    size_t num_labels;             // Number of entries in the labels array.
    char* label_file_data = NULL;  // Buffer holding the complete collection of label strings.

    parse_labels(&labels, &label_file_data, args.labels_file, &num_labels);

    syslog(LOG_INFO, "Start fetching video frames from VDO");
    if (!img_provider_start(image_provider)) {
        panic("%s: Could not start image provider", __func__);
    }

    while (running) {
        struct timeval start_ts, end_ts;
        unsigned int preprocessing_ms = 0;
        unsigned int inference_ms     = 0;
        unsigned int total_elapsed_ms = 0;

        g_autoptr(VdoBuffer) vdo_buf = img_provider_get_frame(image_provider);
        if (!vdo_buf) {
            // This can only happen if it is global rotation then
            // the stream has to be restarted because rotation has been changed.
            syslog(
                LOG_INFO,
                "No buffer because of changed global rotation. Application needs to be restarted");
            goto end;
        }
        // If needed convert and scale/crop to correct input format and resolution
        // Its up to the model provider to decide if needed or not
        // If not needed the model_run_preprocessing will return true without
        // any work
        gettimeofday(&start_ts, NULL);
        if (!model_run_preprocessing(model_provider, vdo_buf)) {
            // No power
            if (!vdo_stream_buffer_unref(image_provider->vdo_stream, &vdo_buf, &vdo_error)) {
                if (!vdo_error_is_expected(&vdo_error)) {
                    panic("%s: Unexpexted error: %s", __func__, vdo_error->message);
                }
                g_clear_error(&vdo_error);
            }
            img_provider_flush_all_frames(image_provider);
            continue;
        }
        gettimeofday(&end_ts, NULL);

        preprocessing_ms = (unsigned int)(((end_ts.tv_sec - start_ts.tv_sec) * 1000) +
                                          ((end_ts.tv_usec - start_ts.tv_usec) / 1000));
        syslog(LOG_INFO, "Ran pre-processing for %u ms", preprocessing_ms);

        // Retrieve detections from data
        gettimeofday(&start_ts, NULL);
        if (!model_run_inference(model_provider, vdo_buf)) {
            // No power
            if (!vdo_stream_buffer_unref(image_provider->vdo_stream, &vdo_buf, &vdo_error)) {
                if (!vdo_error_is_expected(&vdo_error)) {
                    panic("%s: Unexpexted error: %s", __func__, vdo_error->message);
                }
                g_clear_error(&vdo_error);
            }
            img_provider_flush_all_frames(image_provider);
            continue;
        }
        gettimeofday(&end_ts, NULL);

        inference_ms = (unsigned int)(((end_ts.tv_sec - start_ts.tv_sec) * 1000) +
                                      ((end_ts.tv_usec - start_ts.tv_usec) / 1000));
        syslog(LOG_INFO, "Ran inference for %u ms", inference_ms);

        total_elapsed_ms = inference_ms + preprocessing_ms;

        // Check if the framerate from vdo should be changed
        img_provider_update_framerate(image_provider, total_elapsed_ms);

        for (size_t i = 0; i < number_output_tensors; i++) {
            if (!model_get_tensor_output_info(model_provider, i, &tensor_outputs[i])) {
                panic("Failed to get output tensor info for %zu", i);
            }
        }

        uint8_t* tensor_data = tensor_outputs[0].data;
        // Parse the output
        gettimeofday(&start_ts, NULL);

        char plate_string[39] = {0};
        parse_licence_plate(tensor_data,
                            model_params,
                            labels,
                            num_labels,
                            plate_string,
                            sizeof(plate_string));

        syslog(LOG_INFO, "License Plate Text: %s", plate_string);

        gettimeofday(&end_ts, NULL);
        syslog(LOG_INFO, "Ran parsing for %u ms", elapsed_ms(&start_ts, &end_ts));

        // This will allow vdo to fill this buffer with data again
        if (!vdo_stream_buffer_unref(image_provider->vdo_stream, &vdo_buf, &vdo_error)) {
            if (!vdo_error_is_expected(&vdo_error)) {
                panic("%s: Unexpexted error: %s", __func__, vdo_error->message);
            }
            g_clear_error(&vdo_error);
        }
    }

end:
    // Cleanup
    free(model_params);
    if (image_provider) {
        destroy_img_provider(image_provider);
    }
    if (model_provider) {
        destroy_model_provider(model_provider);
    }
    free(tensor_outputs);
    free(labels);
    free(label_file_data);

    syslog(LOG_INFO, "Exit %s", argv[0]);

    return 0;
}
