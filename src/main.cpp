#include <stdio.h>
#include "mnist_model.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "sampleDigits/sampleDigits.h"
#include <ctime>
#include <unistd.h>
long  millis()
{
    return std::clock()/1000;
}
namespace
{
  tflite::ErrorReporter *error_reporter = nullptr;
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 70 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
} 
void initTFInterpreter(){
static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  //Create Model
  model = tflite::GetModel(mnist_model);
  //Verify Version of Tf Micro matches Model's verson
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // the all ops resolver has all ops MicroMutableOpResolver should normal be used to save space from unneed ops
  tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize,
      error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
  error_reporter->Report("Input Shape");
  error_reporter->Report(TfLiteTypeGetName(input->type));
  error_reporter->Report("Output Shape");
  TfLiteTensor *output = interpreter->output(0);
  error_reporter->Report(TfLiteTypeGetName(output->type));
  error_reporter->Report("Arena Used:%d bytes of memory", interpreter->arena_used_bytes());

}


///Returns the index of the max value
uint oneHotDecode(TfLiteTensor *layer){
  int max = 0;
  uint result = 0;
  for (uint i = 0; i < 10; i++)
  {
    // error_reporter->Report("num:%d score:%d", i,
    //                     output->data.int8[i]);
    if (layer->data.int8[i] > max)
    {
      result = i;
      max = layer->data.int8[i];
    }
  }
  return result;
}

int8_t uint8GrayscaleIint8(uint8_t uint8color){
 return (int8_t)(((int)uint8color) - 128); //is there a better way?
}
uint8_t int8GrayscaleUint8(int8_t int8color){
 return (uint8_t)(((int)int8color) + 128); //is there a better way?
}

uint inferNumberImage(int8_t* mnistimage){
memcpy(input->data.int8, mnistimage, 28 * 28);
  for (int i = 0; i < 28 * 28; i++)
  {
    input->data.int8[i] = mnistimage[i];
  }
  long  start = millis();
  if (kTfLiteOk != interpreter->Invoke())//Any error i have in invoke tend to just crash the whole system so i dont usually see this message
  {
    error_reporter->Report("Invoke failed.");
  }
  else
  {
    error_reporter->Report("Invoke passed.");
    error_reporter->Report(" Took :");
    error_reporter->Report(std::to_string(millis()-start).c_str());
    error_reporter->Report(" milliseconds");

  }
  
  TfLiteTensor *output = interpreter->output(0);
  uint result= oneHotDecode(output);
  return result;
}

void testPreloadedNumbers(){
  uint num=inferNumberImage(number1Sample);
  error_reporter->Report("Testing One. Result:");
  error_reporter->Report(std::to_string(num).c_str());
  num=inferNumberImage(number2Sample);
  error_reporter->Report("Testing two. Result:");
  error_reporter->Report(std::to_string(num).c_str());
  num=inferNumberImage(number4Sample);
  error_reporter->Report("Testing four. Result:");
  error_reporter->Report(std::to_string(num).c_str());
  num=inferNumberImage(number5Sample);
  error_reporter->Report("Testing five. Result:");
  error_reporter->Report(std::to_string(num).c_str());
  num=inferNumberImage(number8Sample);
  error_reporter->Report("Testing eight. Result:");
  error_reporter->Report(std::to_string(num).c_str());
  num=inferNumberImage(number9Sample);
  error_reporter->Report("Testing nine. Result:");
  error_reporter->Report(std::to_string(num).c_str());
}

int main()
{
    initTFInterpreter();
    error_reporter->Report("Hello");
    testPreloadedNumbers();

    return 0;
}
