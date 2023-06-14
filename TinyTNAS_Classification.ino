#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>

#include "model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"


namespace {
  const float accelerationThreshold = 0.6; // threshold of significant in G's
  const int n_samples = 100;
  const int n_features = 6;
  int read_samples = n_samples;

  // A buffer holding the last 600 sets of 3-channel values from the accelerometer.
  constexpr int acceleration_data_length = 100 * 3;
  float acceleration_data[acceleration_data_length] = {};
  int acceleration_data_index = 0;
  float acceleration_sample_rate = 0.0f;

  // A buffer holding the last 600 sets of 3-channel values from the gyroscope.
  constexpr int gyroscope_data_length = 100 * 3;
  float gyroscope_data[gyroscope_data_length] = {};
  int gyroscope_data_index = 0;
  float gyroscope_sample_rate = 0.0f;  

  constexpr int kTensorArenaSize = 60 * 1024;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];

  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;

  constexpr int label_count = 4;
  const char* labels[label_count] = {"badminton", "running", "standing", "walking"}; // [0:badminton, 1:running, 2:standing, 3:walking]

  void SetupIMU() {
    // Make sure we are pulling measurements into a FIFO.
    // If you see an error on this line, make sure you have at least v1.1.0 of the
    // Arduino_LSM9DS1 library installed.
    IMU.setContinuousMode();
    acceleration_sample_rate = IMU.accelerationSampleRate();
    gyroscope_sample_rate = IMU.gyroscopeSampleRate();
  }

}

void setup() {
  tflite::InitializeTarget();  // setup serial port
  MicroPrintf("Started");

  if (!IMU.begin()) {
    MicroPrintf("Failed to initialized IMU!");
    while (true) {
      // NO RETURN
    }
  }
  SetupIMU();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // pull in all the TFLM ops, you can remove this line and
  // only pull in the TFLM ops you need, if would like to reduce
  // the compiled size of the sketch.
  static tflite::AllOpsResolver op_resolver;  
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();  

  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != n_samples) ||
      (model_input->dims->data[2] != n_features) ||
      (model_input->type != kTfLiteInt8) ||
      (model_input->params.zero_point != -128) ||
      (model_input->params.scale != 1.0)) {
    MicroPrintf("Bad input tensor parameters in model");
    return;
  }
  
  TfLiteTensor* model_output = interpreter->output(0);
  if ((model_output->dims->size != 2) || (model_output->dims->data[0] != 1) ||
      (model_output->dims->data[1] != label_count) ||
      (model_output->type != kTfLiteInt8)) {
    MicroPrintf("Bad output tensor parameters in model");
    return;
  }
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;
  static bool is_initialized = false;
  if (!is_initialized) {
    pinMode(LED_BUILTIN, OUTPUT);
    // Pins for the built-in RGB LEDs on the Arduino Nano 33 BLE Sense
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    // Ensure the LED is off by default.
    // Note: The RGB LEDs on the Arduino Nano 33 BLE
    // Sense are on when the pin is LOW, off when HIGH.
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
    digitalWrite(LED_BUILTIN, HIGH);

    is_initialized = true;
  }    

  while (read_samples == n_samples) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);
      // check if it's above the threshold
      if (aSum >= accelerationThreshold) {
        // reset the sample read count
        read_samples = 0;
        digitalWrite(LEDR, HIGH);
        digitalWrite(LEDG, HIGH);
        digitalWrite(LEDB, HIGH);
        digitalWrite(LED_BUILTIN, HIGH);
        break;
      }
    }
  }

  // check if the all the required samples have been read since
  // the last time the significant motion was detected
  TfLiteTensor* model_input = interpreter->input(0);
  while (read_samples < n_samples) {
    // check if new acceleration AND gyroscope data is available
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // read the acceleration and gyroscope data
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);    

      // normalize the IMU data between 0 to 1 and store in the model's
      // input tensor
      model_input->data.int8[read_samples * 6 + 1] = aY;
      model_input->data.int8[read_samples * 6 + 0] = aX;
      model_input->data.int8[read_samples * 6 + 2] = aZ;
      model_input->data.int8[read_samples * 6 + 3] = gX;
      model_input->data.int8[read_samples * 6 + 4] = gY;
      model_input->data.int8[read_samples * 6 + 5] = gZ;
      
      read_samples++;
      if (read_samples == n_samples) {
        // Run inferencing
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
          MicroPrintf("Invoke failed");
          return;
        }
        
        TfLiteTensor* output = interpreter->output(0);

        int8_t max_score;
        int max_index;
        for (int i = 0; i < label_count; ++i) {
          const int8_t score = output->data.int8[i];
          if ((i == 0) || (score > max_score)) {
            max_score = score;
            max_index = i;
          }
        }

        float max_score_f = (max_score - output->params.zero_point) * output->params.scale;
        float max_score_int;
        float max_score_frac = modf(max_score_f * 100, &max_score_int);
        
        MicroPrintf("Found %s (%d.%d%%)", labels[max_index], static_cast<int>(max_score_int), static_cast<int>(max_score_frac * 100));

        if (labels[max_index] == "walking") {
          digitalWrite(LEDG, LOW);  // Green for walking
        } else if (labels[max_index] == "badminton") {
          digitalWrite(LEDR, LOW);  // Red for badminton
        } else if (labels[max_index] == "standing") {
          digitalWrite(LEDB, LOW);  // Blue for standing
        } else {
          digitalWrite(LED_BUILTIN, LOW); // running
        }
      }
    }
  }
}
