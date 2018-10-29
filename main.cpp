#include <iostream>

#include "tensorflow/contrib/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"

//#include "tensorflow/contrib/lite/stderr_reporter.h"
//#include "tensorflow/contrib/lite/core/api/error_reporter.h"

int main() {

  // tflite::ErrorReporter* error_reporter = tflite::DefaultErrorReporter();
  // tflite::FlatBufferModel model("model.tflite", *error_reporter);

  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile("model.tflite");

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder builder(*model, resolver);

  std::unique_ptr<tflite::Interpreter> interpreter;

  // Build interpreter
  if (builder(&interpreter) == kTfLiteOk) {
    // Fill allocate tensors
    if (interpreter->AllocateTensors() == kTfLiteOk) {

      float *input = interpreter->typed_input_tensor<float>(0);
      float *output = interpreter->typed_output_tensor<float>(0);

      // Populate input
      input[0] = 5.1;
      std::cout << input[0] << std::endl;
      interpreter->Invoke();
      std::cout << output[0] << std::endl;

      // Populate input second time
      input[0] = -3;
      std::cout << input[0] << std::endl;
      interpreter->Invoke();
      std::cout << output[0] << std::endl;
    }
  }

  return 0;
}
