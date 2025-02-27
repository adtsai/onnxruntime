// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/util/math.h"
#include <algorithm>
#include <cmath>

namespace onnxruntime {
namespace test {

TEST(MathOpTest, Add_int32) {
  OpTester test("Add");
  test.AddInput<int32_t>("A", {3}, {1, 2, 3});
  test.AddInput<int32_t>("B", {3}, {4, 5, 6});
  test.AddOutput<int32_t>("C", {3}, {5, 7, 9});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT parser: elementwise inputs must not be Int32
}

TEST(MathOpTest, Add_int64) {
  OpTester test("Add");
  test.AddInput<int64_t>("A", {3}, {1, 2, 3});
  test.AddInput<int64_t>("B", {3}, {4, 5, 6});
  test.AddOutput<int64_t>("C", {3}, {5, 7, 9});
  test.Run();
}

TEST(MathOpTest, Add_float) {
  OpTester test("Add");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, -1.0f,
                        0.0f, 1.5f, -100.0f,
                        -5.4f, 9.3f, -10'000.0f});
  test.AddInput<float>("B", dims,
                       {-1.0f, 4.4f, 432.3f,
                        0.0f, 3.5f, 64.0f,
                        -5.4f, 9.3f, 10'000.0f});
  test.AddOutput<float>("C", dims,
                        {0.0f, 6.4f, 431.3f,
                         0.0f, 5.0f, -36.0f,
                         -10.8f, 18.6f, 0.0f});

#if defined(OPENVINO_CONFIG_MYRIAD) || defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_VAD_M)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  //OpenVINO: Disabled due to accuracy mismatch for FP16
#else
  test.Run();
#endif
}

TEST(MathOpTest, Add_double) {
  OpTester test("Add");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<double>("A", dims,
                        {1.0, 2.0, -1.0,
                         0.0, 1.5, -100.0,
                         -5.4, 9.3, -10'000.0});
  test.AddInput<double>("B", dims,
                        {-1.0, 4.4, 432.3,
                         0.0, 3.5, 64.0,
                         -5.4, 9.3, 10'000.0});
  test.AddOutput<double>("C", dims,
                         {0.0, 6.4, 431.3,
                          0.0, 5.0, -36.0,
                          -10.8, 18.6, 0.0});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  // Disabling OpenVINO as this type is not supported
}

TEST(MathOpTest, Add_Broadcast_Axis) {
  OpTester test("Add");

  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f,
                        7.0f, 8.0f, 9.0f});
  test.AddInput<float>("B", {3, 1},
                       {3.0f,
                        2.0f,
                        1.0f});
  test.AddOutput<float>("C", dims,
                        {4.0f, 5.0f, 6.0f,
                         6.0f, 7.0f, 8.0f,
                         8.0f, 9.0f, 10.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "");
}

TEST(MathOpTest, Add_Broadcast_0x0) {
  OpTester test("Add");

  test.AddInput<float>("A", {}, {10.0f});
  test.AddInput<float>("B", {}, {2.0f});
  test.AddOutput<float>("C", {}, {12.0f});
  test.Run();
}

TEST(MathOpTest, Add_Broadcast_0x1) {
  OpTester test("Add");

  test.AddInput<float>("A", {}, {10.0f});
  test.AddInput<float>("B", {1}, {2.0f});
  test.AddOutput<float>("C", {1}, {12.0f});
  test.Run();
}

TEST(MathOpTest, Add_Broadcast_1x0) {
  OpTester test("Add");

  test.AddInput<float>("A", {1}, {10.0f});
  test.AddInput<float>("B", {}, {2.0f});
  test.AddOutput<float>("C", {1}, {12.0f});
  test.Run();
}

TEST(MathOpTest, Add_Broadcast_1x1) {
  OpTester test("Add");

  test.AddInput<float>("A", {1}, {10.0f});
  test.AddInput<float>("B", {1}, {2.0f});
  test.AddOutput<float>("C", {1}, {12.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "");
}

TEST(MathOpTest, Add_Broadcast_3x2_3x1) {
  OpTester test("Add");

  std::vector<int64_t> dims{3, 2};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f});
  test.AddInput<float>("B", {3, 1},
                       {1.0f,
                        2.0f,
                        3.0f});
  test.AddOutput<float>("C", dims,
                        {2.0f, 3.0f,
                         5.0f, 6.0f,
                         8.0f, 9.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "");
}

TEST(MathOpTest, Add_Broadcast_2x1x4_1x3x1) {
  OpTester test("Add");

  test.AddInput<float>("A", {2, 1, 4},
                       {101.0f, 102.0f, 103.0f, 104.0f,
                        201.0f, 202.0f, 203.0f, 204.0f});
  test.AddInput<float>("B", {1, 3, 1},
                       {010.0f, 020.0f, 030.0f});
  test.AddOutput<float>("C", {2, 3, 4},
                        {111.0f, 112.0f, 113.0f, 114.0f,
                         121.0f, 122.0f, 123.0f, 124.0f,
                         131.0f, 132.0f, 133.0f, 134.0f,

                         211.0f, 212.0f, 213.0f, 214.0f,
                         221.0f, 222.0f, 223.0f, 224.0f,
                         231.0f, 232.0f, 233.0f, 234.0f});

#if defined(OPENVINO_CONFIG_MYRIAD) || defined(OPENVINO_CONFIG_VAD_M)
  //OpenVINO: Disabled due to software limitation for VPU Plugin.
  //This test runs fine on CPU and GPU Plugins
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Input batch size is inconsistent
#endif
}

TEST(MathOpTest, Add_Broadcast_2x1x1_3x4) {
  OpTester test("Add");

  test.AddInput<float>("A", {2, 1, 1},
                       {100.0f, 200.0f});
  test.AddInput<float>("B", {3, 4},
                       {011.0f, 012.0f, 013.0f, 014.0f,
                        021.0f, 022.0f, 023.0f, 024.0f,
                        031.0f, 032.0f, 033.0f, 034.0f});
  test.AddOutput<float>("C", {2, 3, 4},
                        {111.0f, 112.0f, 113.0f, 114.0f,
                         121.0f, 122.0f, 123.0f, 124.0f,
                         131.0f, 132.0f, 133.0f, 134.0f,

                         211.0f, 212.0f, 213.0f, 214.0f,
                         221.0f, 222.0f, 223.0f, 224.0f,
                         231.0f, 232.0f, 233.0f, 234.0f});
#if defined(OPENVINO_CONFIG_MYRIAD) || defined(OPENVINO_CONFIG_VAD_M)
  //OpenVINO: Disabled due to software limitation for VPU Plugin.
  //This test runs fine on CPU and GPU Plugins
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Input batch size is inconsistent
#endif
}

// Validate runtime failure has useful error message when ORT_ENFORCE is used
TEST(MathOpTest, Add_Invalid_Broadcast) {
  OpTester test("Add");

  std::vector<int64_t> dims{2, 3};

  // Use symbolic dimension for first dim so it doesn't fail during shape inferencing
  test.AddShapeToTensorData(true, 0);

  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f});
  test.AddInput<float>("B", {3, 1},
                       {1.0f,
                        2.0f,
                        3.0f});
  test.AddOutput<float>("C", dims,
                        {0.0f, 0.0f,
                         0.0f, 0.0f,
                         0.0f, 0.0f});

  // Call Run twice to validate different parts of the error message.
  // Only test on CPU as it's that implementation that has the ORT_ENFORCE we're targeting
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());

  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Non-zero status code returned while running Add node. Name:'node1'",
           {}, nullptr, &execution_providers);

  // test.Run std::move's the EP from execution_providers into the per-Run session so need to re-create
  execution_providers[0] = DefaultCpuExecutionProvider();
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "axis == 1 || axis == largest was false. "
           "Attempting to broadcast an axis by a dimension other than 1. 2 by 3",
           {}, nullptr, &execution_providers);
}

TEST(MathOpTest, Sub_int32) {
  OpTester test("Sub");
  test.AddInput<int32_t>("A", {3}, {1, 4, 3});
  test.AddInput<int32_t>("B", {3}, {4, 2, 4});
  test.AddOutput<int32_t>("C", {3}, {-3, 2, -1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT parser:elementwise inputs must not be Int32
}

TEST(MathOpTest, Sub_int64) {
  OpTester test("Sub");
  test.AddInput<int64_t>("A", {3}, {1, 5, 6});
  test.AddInput<int64_t>("B", {3}, {4, 5, 3});
  test.AddOutput<int64_t>("C", {3}, {-3, 0, 3});
  test.Run();
}

TEST(MathOpTest, Sub) {
  OpTester test("Sub");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, -1.0f,
                        0.0f, 1.5f, -100.0f,
                        -5.4f, 9.3f, -10'000.0f});
  test.AddInput<float>("B", dims,
                       {-1.0f, 4.4f, 432.3f,
                        0.0f, 3.5f, 64.0f,
                        -5.4f, 9.3f, 10'000.0f});
  test.AddOutput<float>("C", dims,
                        {2.0f, -2.4f, -433.3f,
                         0.0f, -2.0f, -164.0f,
                         0.0f, 0.0f, -20'000.0f});
  test.Run();
}

TEST(MathOpTest, Sub_Broadcast_Scalar) {
  OpTester test("Sub");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, -1.0f,
                        0.0f, 1.5f, -100.0f,
                        -5.4f, 9.3f, -10'000.0f});
  test.AddInput<float>("B", {}, {5.0f});
  test.AddOutput<float>("C", dims,
                        {-4.0f, -3.0f, -6.0f,
                         -5.0f, -3.5f, -105.0f,
                         -10.4f, 4.3f, -10'005.0f});
  test.Run();
}

TEST(MathOpTest, Mul_int32) {
  OpTester test("Mul");
  test.AddInput<int32_t>("A", {3}, {1, 2, 3});
  test.AddInput<int32_t>("B", {3}, {4, -3, 6});
  test.AddOutput<int32_t>("C", {3}, {4, -6, 18});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT parser:elementwise inputs must not be Int32
}

TEST(MathOpTest, Mul_int64) {
  OpTester test("Mul");
  test.AddInput<int64_t>("A", {3}, {3, 6, -3});
  test.AddInput<int64_t>("B", {3}, {4, -3, -2});
  test.AddOutput<int64_t>("C", {3}, {12, -18, 6});
  test.Run();
}

TEST(MathOpTest, Mul) {
  OpTester test("Mul");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, -1.0f,
                        0.0f, 1.5f, -100.0f, -5.4f,
                        9.30f, -10'000.0f});
  test.AddInput<float>("B", dims,
                       {-1.0f, 4.4f, 432.3f,
                        0.0f, 3.5f, 64.0f, -5.4f,
                        9.30f, 10'000.0f});
  test.AddOutput<float>("C", dims,
                        {-1.0f, 8.8f, -432.3f,
                         0.0f, 5.25f, -6'400.0f,
                         29.16f, 86.49f, -100'000'000.0f});

#if defined(OPENVINO_CONFIG_MYRIAD) || defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_VAD_M)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  //OpenVINO: Disabled due to accuracy issues for MYRIAD FP16
#else
  test.Run();
#endif
}

TEST(MathOpTest, Div_int32) {
  OpTester test("Div");
  test.AddInput<int32_t>("A", {3}, {4, 8, 8});
  test.AddInput<int32_t>("B", {3}, {1, 3, 2});
  test.AddOutput<int32_t>("C", {3}, {4, 2, 4});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT parser:elementwise inputs must not be Int32
}

TEST(MathOpTest, Div_int64) {
  OpTester test("Div");
  test.AddInput<int64_t>("A", {3}, {4, 8, 8});
  test.AddInput<int64_t>("B", {3}, {2, 3, 4});
  test.AddOutput<int64_t>("C", {3}, {2, 2, 2});
  test.Run();
}

TEST(MathOpTest, Div) {
  OpTester test("Div");
  std::vector<int64_t> dims{2, 3};
  test.AddInput<float>("A", dims,
                       {1'000.0f, 1.0f, 6.0f,
                        0.0f, -10.0f, -1.0f});
  test.AddInput<float>("B", dims,
                       {1'000.0f, 2.0f, 3.0f,
                        1.0f, -1.0f, 4.0f});
  test.AddOutput<float>("C", dims,
                        {1.0f, 0.5f, 2.0f,
                         0.0f, 10.0f, -0.25f});
#if defined(OPENVINO_CONFIG_GPU_FP32) || defined(OPENVINO_CONFIG_GPU_FP16)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  //OpenVINO: Will be enabled in the next release
#else
  test.Run();
#endif
}

TEST(MathOpTest, Abs) {
  OpTester test("Abs");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, -2.0f, -0.0f, -10.0f});
  test.AddOutput<float>("Y", dims, {1.0f, 2.0f, 0.0f, 10.0f});
  test.Run();
}

TEST(MathOpTest, Abs_int8) {
  OpTester test("Abs");
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("X", dims, {1, 2, -1, -5});
  test.AddOutput<int8_t>("Y", dims, {1, 2, 1, 5});
  test.Run();
}

TEST(MathOpTest, Abs_int32) {
  OpTester test("Abs");
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("X", dims, {1, 2, -1, -5});
  test.AddOutput<int32_t>("Y", dims, {1, 2, 1, 5});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT parser: Int32 not allowed as input to this layer
}

TEST(MathOpTest, Neg) {
  OpTester test("Neg");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {1.0f, -2.0f,
                        0.0f, -10.0f});
  test.AddOutput<float>("Y", dims,
                        {-1.0f, 2.0f,
                         -0.0f, 10.0f});
  test.Run();
}

TEST(MathOpTest, Neg_int8) {
  OpTester test("Neg");
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("X", dims, {1, -2, 0, -10});
  test.AddOutput<int8_t>("Y", dims, {-1, 2, 0, 10});
  test.Run();
}

TEST(MathOpTest, Neg_int32) {
  OpTester test("Neg");
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("X", dims, {1, -2, 0, -10});
  test.AddOutput<int32_t>("Y", dims, {-1, 2, 0, 10});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT parser: Int32 not allowed as input to this layer
}

TEST(MathOpTest, Floor) {
  OpTester test("Floor");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {-1.5f, 0.2f,
                        -0.5f, 10.3f});
  test.AddOutput<float>("Y", dims,
                        {-2.0f, 0.0f,
                         -1.0f, 10.0f});
  test.Run();
}

TEST(MathOpTest, Ceil) {
  OpTester test("Ceil");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {-1.5f, 0.2f,
                        -0.5f, 10.3f});
  test.AddOutput<float>("Y", dims,
                        {-1.0f, 1.0f,
                         0.0f, 11.0f});
  test.Run();
}

TEST(MathOpTest, Reciprocal) {
  OpTester test("Reciprocal");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {1.0f, 2.0f,
                        -1.0f, -2.0f});
  test.AddOutput<float>("Y", dims,
                        {1.0f, 0.5f,
                         -1.0f, -0.5f});
  test.Run();
}

TEST(MathOpTest, Sqrt_Float) {
  OpTester test("Sqrt");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {1.0f, 4.0f,
                        0.0f, 9.0f});
  test.AddOutput<float>("Y", dims,
                        {1.0f, 2.0f,
                         0.0f, 3.0f});
  test.Run();
}

TEST(MathOpTest, Sqrt_Double) {
  OpTester test("Sqrt");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<double>("X", dims,
                        {1.0, 4.0,
                         0.0, 9.0});
  test.AddOutput<double>("Y", dims,
                         {1.0, 2.0,
                          0.0, 3.0});
  test.Run();
}

TEST(MathOpTest, Pow_Float) {
  OpTester test("Pow");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {2.0f, 2.0f,
                        std::sqrt(2.0f), 1.0f});
  test.AddInput<float>("Y", dims,
                       {0.0f, 8.0f,
                        2.0f, 9.0f});
  test.AddOutput<float>("Z", dims,
                        {1.0f, 256.0f,
                         2.0f, 1.0f});
  test.Run();
}

TEST(MathOpTest, Pow_Double) {
  OpTester test("Pow");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<double>("X", dims,
                        {2.0, 2.0,
                         std::sqrt(2.0), 1.0});
  test.AddInput<double>("Y", dims,
                        {0.0, 8.0,
                         2.0, 9.0});
  test.AddOutput<double>("Z", dims,
                         {1.0, 256.0,
                          2.0, 1.0});
  test.Run();
}

TEST(MathOpTest, Pow_Broadcast_Scalar0) {
  OpTester test("Pow");

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", {}, {2.0f});
  test.AddInput<float>("Y", dims, {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("Z", dims, {2.0f, 4.0f, 8.0f});
  test.Run();
}

TEST(MathOpTest, Pow_Broadcast_Scalar1) {
  OpTester test("Pow");

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("Y", {}, {2.0f});
  test.AddOutput<float>("Z", dims, {1.0f, 4.0f, 9.0f});
  test.Run();
}

TEST(MathOpTest, Exp_float) {
  OpTester test("Exp");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {0.0f, 1.0f,
                        2.0f, 10.0f});
  test.AddOutput<float>("Y", dims,
                        {1.0f, std::exp(1.0f),
                         std::exp(2.0f), std::exp(10.0f)});
  test.SetOutputRelErr("Y", 1e-7f);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: result differs
}

TEST(MathOpTest, Exp_double) {
  OpTester test("Exp");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<double>("X", dims,
                        {0.0, 1.0,
                         2.0, 10.0});
  test.AddOutput<double>("Y", dims,
                         {1.0, std::exp(1.0),
                          std::exp(2.0), std::exp(10.0)});
  test.SetOutputRelErr("Y", 1e-7f);
  // TODO: Check if this test's result really differs for tensorRT
  // For now basing this exclusion based on this test's float counterpart - Exp_float
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(MathOpTest, Log) {
  OpTester test("Log");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {1.0f, 2.0f,
                        5.0f, 10.0f});
  test.AddOutput<float>("Y", dims,
                        {0.0f, std::log(2.0f),
                         std::log(5.0f), std::log(10.0f)});
  test.Run();
}

TEST(MathOpTest, Sum_6) {
  OpTester test("Sum", 6);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.4f, 0.01f, -10'000.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 64.0f,
                        -1.0f, 0.02f, 0.25f});
  test.AddInput<float>("data_3", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 64.0f,
                        5.4f, 0.03f, 10'000.0f});
  test.AddOutput<float>("sum", dims,
                        {3.0f, 0.0f, 6.0f,
                         -6.0f, 6.6f, 28.0f,
                         -1.0f, 0.06f, 0.25f});

#if defined(OPENVINO_CONFIG_MYRIAD) || defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_VAD_M)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});  //OpenVINO: Disabled due to accuracy mismatch for FP16
#else
  test.Run();
#endif
}

TEST(MathOpTest, Sum_8_Test1) {
  OpTester test("Sum", 8);
  test.AddInput<float>("data_0", {3}, {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("data_1", {3, 1}, {10.0f, 20.0f, 30.0f});
  test.AddInput<float>("data_2", {3, 1, 1}, {100.0f, 200.0f, 300.0f});
  test.AddOutput<float>("sum", {3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,

                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,

                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
#if defined(OPENVINO_CONFIG_MYRIAD) || defined(OPENVINO_CONFIG_VAD_M)
  //OpenVINO: Disabled due to software limitation for VPU Plugin.
  //This test runs fine on CPU and GPU Plugins
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});                    //TensorRT: Expected output shape [{3,3,3}] did not match run output shape [{3,1,1}] for sum
#endif
}

TEST(MathOpTest, Sum_8_Test2) {
  OpTester test("Sum", 8);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {
                           1.0f,
                           0.0f,
                           1.0f,
                           -1.0f,
                           1.1f,
                           -100.0f,
                           -5.4f,
                           0.01f,
                           -74.0f,
                       });
  std::vector<int64_t> dims_1{3};
  test.AddInput<float>("data_1", dims_1,
                       {1.0f, 0.0f, 2.0f});
  std::vector<int64_t> dims_2{3, 1};
  test.AddInput<float>("data_2", dims_2,
                       {-3.0f, 3.3f, 64.0f});
  test.AddOutput<float>("sum", dims,
                        {-1.0f, -3.0f, 0.0f,
                         3.3f, 4.4f, -94.7f,
                         59.6f, 64.01f, -8.0f});

#if defined(OPENVINO_CONFIG_MYRIAD) || defined(OPENVINO_CONFIG_VAD_M)
  //OpenVINO: Disabled due to software limitation for VPU Plugin.
  //This test runs fine on CPU and GPU Plugins
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "Sum is not correct", {kTensorrtExecutionProvider});  //TensorRT: result differs
#endif
}

TEST(MathOpTest, Min_6) {
  OpTester test("Min", 6);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.4f, 0.01f, -10'000.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 64.0f,
                        -1.0f, 0.02f, 0.1f});
  test.AddInput<float>("data_3", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 64.0f,
                        5.4f, 0.03f, 10'000.0f});
  test.AddOutput<float>("sum", dims,
                        {1.0f, 0.0f, 1.0f,
                         -3.0f, 1.1f, -100.0f,
                         -5.4f, 0.01f, -10'000.0f});
  test.Run();
}

TEST(MathOpTest, Min_8) {
  OpTester test("Min", 8);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.4f, 0.01f, -10'000.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 64.0f,
                        -1.0f, 0.02f, 0.1f});
  test.AddInput<float>("data_3", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 64.0f,
                        5.4f, 0.03f, 10'000.0f});
  test.AddOutput<float>("min", dims,
                        {1.0f, 0.0f, 1.0f,
                         -3.0f, 1.1f, -100.0f,
                         -5.4f, 0.01f, -10'000.0f});
  test.Run();
}

TEST(MathOpTest, Max_6) {
  OpTester test("Max", 6);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.4f, 0.01f, -10'000.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 64.0f,
                        -1.0f, 0.02f, 0.1f});
  test.AddInput<float>("data_2", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 64.0f,
                        5.4f, 0.03f, 10'000.0f});
  test.AddOutput<float>("max", dims,
                        {1.0f, 0.0f, 3.0f,
                         -1.0f, 3.3f, 64.0f,
                         5.4f, 0.03f, 10'000.0f});
  test.Run();
}

TEST(MathOpTest, Max_8_Float) {
  OpTester test("Max", 8);
  test.AddInput<float>("data_0", {1, 3},
                       {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("data_2", {3, 3},
                       {10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f,
                        70.0f, 80.0f, 90.0f});
  test.AddInput<float>("data_1", {3, 1},
                       {-1.0f, -2.0f, 300.0f});
  test.AddOutput<float>("max", {3, 3},
                        {10.0f, 20.0f, 30.0f,
                         40.0f, 50.0f, 60.0f,
                         300.0f, 300.0f, 300.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Max_8_Double) {
  OpTester test("Max", 8);
  test.AddInput<double>("data_0", {1, 3},
                        {1.0, 2.0, 3.0});
  test.AddInput<double>("data_2", {3, 3},
                        {10.0, 20.0, 30.0,
                         40.0, 50.0, 60.0,
                         70.0, 80.0, 90.0});
  test.AddInput<double>("data_1", {3, 1},
                        {-1.0, -2.0, 300.0});
  test.AddOutput<double>("max", {3, 3},
                         {10.0, 20.0, 30.0,
                          40.0, 50.0, 60.0,
                          300.0, 300.0, 300.0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Max_8_2inputbroadcast) {
  OpTester test("Max", 8);
  test.AddInput<float>("data_0", {1, 3},
                       {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("data_1", {3, 3},
                       {10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f,
                        70.0f, 80.0f, 90.0f});
  test.AddOutput<float>("max", {3, 3},
                        {10.0f, 20.0f, 30.0f,
                         40.0f, 50.0f, 60.0f,
                         70.0f, 80.0f, 90.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Input batch size is inconsistent
}

TEST(MathOpTest, Not) {
  OpTester test("Not");
  std::vector<int64_t> dims{2};
  test.AddInput<bool>("X", dims, {false, true});
  test.AddOutput<bool>("Y", dims, {true, false});
  test.Run();
}

TEST(MathOpTest, And) {
  OpTester test("And");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {false, false, false, true});
  test.Run();
}

TEST(MathOpTest, Or) {
  OpTester test("Or");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {false, true, true, true});
  test.Run();
}

TEST(MathOpTest, Xor) {
  OpTester test("Xor");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {false, true, true, false});
  test.Run();
}

TEST(MathOpTest, Xor_bcast3v2d) {
  OpTester test("Xor");

  test.AddInput<bool>("A", {2, 3, 4},
                      {false, true, false, true,
                       false, true, false, true,
                       false, true, false, true,

                       false, true, false, true,
                       false, true, false, true,
                       false, true, false, true});
  test.AddInput<bool>("B", {3, 4},
                      {false, false, true, true,
                       false, false, true, true,
                       false, false, true, true});
  test.AddOutput<bool>("C", {2, 3, 4},
                       {false, true, true, false,
                        false, true, true, false,
                        false, true, true, false,

                        false, true, true, false,
                        false, true, true, false,
                        false, true, true, false});
  test.Run();
}

TEST(MathOpTest, Less) {
  OpTester test("Less");
  std::vector<int64_t> dims{4};
  test.AddInput<float>("A", dims, {1.0f, 0.0f, -1.0f, -1.0f});
  test.AddInput<float>("B", dims, {1.0f, 1.0f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", dims, {false, true, true, false});
  test.Run();
}

TEST(MathOpTest, Less_Scalar0) {
  OpTester test("Less");
  test.AddInput<float>("A", {1}, {1.0f});
  test.AddInput<float>("B", {4}, {1.0f, 1.5f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", {4}, {false, true, true, false});
  test.Run();
}

TEST(MathOpTest, Less_Scalar1) {
  OpTester test("Less");
  test.AddInput<float>("A", {4}, {1.0f, 0.5f, 2.0f, -1.0f});
  test.AddInput<float>("B", {1}, {1.0f});
  test.AddOutput<bool>("C", {4}, {false, true, false, true});
  test.Run();
}

TEST(MathOpTest, Less_int64_Scalar1) {
  OpTester test("Less", 9);
  test.AddInput<int64_t>("A", {4}, {1, 0, 2, -1});
  test.AddInput<int64_t>("B", {1}, {1});
  test.AddOutput<bool>("C", {4}, {false, true, false, true});
  test.Run();
}

TEST(MathOpTest, Greater_7) {
  OpTester test("Greater");
  std::vector<int64_t> dims{4};
  test.AddInput<float>("A", dims, {1.0f, 0.0f, -1.0f, -1.0f});
  test.AddInput<float>("B", dims, {1.0f, 1.0f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", dims, {false, false, false, false});
  test.Run();
}

TEST(MathOpTest, Greater_9_float) {
  OpTester test("Greater", 9);
  std::vector<int64_t> dims{4};
  test.AddInput<float>("A", dims, {1.0f, 0.0f, -1.0f, -1.0f});
  test.AddInput<float>("B", dims, {1.0f, 1.0f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", dims, {false, false, false, false});
  test.Run();
}

TEST(MathOpTest, Greater_9_int32) {
  OpTester test("Greater", 9);
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<int32_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, false, true});
  test.Run();
}

TEST(MathOpTest, Greater_9_int64) {
  OpTester test("Greater", 9);
  std::vector<int64_t> dims{4};
  test.AddInput<int64_t>("A", dims, {10, 11, 12, 13});
  test.AddInput<int64_t>("B", dims, {15, 7, 12, 9});
  test.AddOutput<bool>("C", dims, {false, true, false, true});
  test.Run();
}

TEST(MathOpTest, Equal_bool) {
  OpTester test("Equal");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run();
}

TEST(MathOpTest, Equal_bool_scalar0) {
  OpTester test("Equal");
  test.AddInput<bool>("A", {1}, {false});
  test.AddInput<bool>("B", {4}, {false, false, true, true});
  test.AddOutput<bool>("C", {4}, {true, true, false, false});
  test.Run();
}

TEST(MathOpTest, Equal_bool_scalar1) {
  OpTester test("Equal");
  test.AddInput<bool>("A", {4}, {false, false, true, true});
  test.AddInput<bool>("B", {1}, {false});
  test.AddOutput<bool>("C", {4}, {true, true, false, false});
  test.Run();
}

TEST(MathOpTest, Equal_int32) {
  OpTester test("Equal");
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("A", dims, {1, 0, -1, -1});
  test.AddInput<int32_t>("B", dims, {1, 1, 2, -1});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run();
}

TEST(MathOpTest, Equal_int64) {
  OpTester test("Equal");
  std::vector<int64_t> dims{4};
  test.AddInput<int64_t>("A", dims, {1, 0, -1, -1});
  test.AddInput<int64_t>("B", dims, {1, 1, 2, -1});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run();
}

TEST(MathOpTest, Equal_float) {
  OpTester test("Equal", 11);
  std::vector<int64_t> dims{4};
  test.AddInput<float>("A", dims, {1.0f, 0.0f, -1.0f, -1.0f});
  test.AddInput<float>("B", dims, {1.0f, 1.0f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run();
}

TEST(MathOpTest, Mean_6) {
  OpTester test("Mean", 6);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.0f, 0.01f, -10.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 65.0f,
                        -1.0f, 0.02f, -1.0f});
  test.AddInput<float>("data_3", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 65.0f,
                        -3.0f, 0.03f, -1.0f});
  test.AddOutput<float>("mean", dims,
                        {1.0f, 0.0f, 2.0f,
                         -2.0f, 2.2f, 10.0f,
                         -3.0f, 0.02f, -4.0f});
  test.Run();
}

TEST(MathOpTest, Mean_8) {
  OpTester test("Mean", 8);
  test.AddInput<float>("data_0", {1}, {1.0f});
  test.AddInput<float>("data_1", {3, 1},
                       {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("data_3", {3, 3},
                       {10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f,
                        70.0f, 80.0f, 90.0f});
  test.AddOutput<float>("mean", {3, 3},
                        {12.0f / 3.0f, 22.0f / 3.0f, 32.0f / 3.0f,
                         43.0f / 3.0f, 53.0f / 3.0f, 63.0f / 3.0f,
                         74.0f / 3.0f, 84.0f / 3.0f, 94.0f / 3.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Input batch size is inconsistent
}

template <float (&op)(float value)>
void TrigFloatTest(OpTester& test, std::initializer_list<float> input) {
  std::vector<int64_t> dims{static_cast<int64_t>(input.size())};

  std::vector<float> output;
  for (auto v : input)
    output.push_back(op(v));

  test.AddInput<float>("X", dims, input);
  test.AddOutput<float>("Y", dims, output);
  test.Run();
}

template <double (&op)(double value)>
void TrigDoubleTest(OpTester& test, std::initializer_list<double> input) {
  std::vector<int64_t> dims{static_cast<int64_t>(input.size())};

  std::vector<double> output;
  for (auto v : input)
    output.push_back(op(v));

  test.AddInput<double>("X", dims, input);
  test.AddOutput<double>("Y", dims, output);
  test.Run();
}

TEST(MathOpTest, SinFloat) {
  OpTester test("Sin");
  TrigFloatTest<std::sin>(test, {1.1f, -1.1f, 2.2f, -2.2f});
}

TEST(MathOpTest, SinDouble) {
  OpTester test("Sin");
  TrigDoubleTest<std::sin>(test, {1.1, -1.1, 2.2, -2.2});
}

TEST(MathOpTest, Cos) {
  OpTester test("Cos");
  TrigFloatTest<std::cos>(test, {1.1f, -1.1f, 2.2f, -2.2f});
}

TEST(MathOpTest, Tan) {
  OpTester test("Tan");
  TrigFloatTest<std::tan>(test, {-100.0f, -50.0f, 0.0f, 50.0f, 100.0f});
}

TEST(MathOpTest, Asin) {
  OpTester test("Asin");
  TrigFloatTest<std::asin>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Acos) {
  OpTester test("Acos");
  TrigFloatTest<std::acos>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Atan) {
  OpTester test("Atan");
  TrigFloatTest<std::atan>(test, {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f});
}

TEST(MathOpTest, Sinh) {
  OpTester test("Sinh", 9);
  TrigFloatTest<std::sinh>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Cosh) {
  OpTester test("Cosh", 9);
  TrigFloatTest<std::cosh>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Asinh) {
  OpTester test("Asinh", 9);
  TrigFloatTest<std::asinh>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Acosh) {
  OpTester test("Acosh", 9);
  TrigFloatTest<std::acosh>(test, {1.0f, 1.1f, 3.0f, 10.0f, 100.0f});
}

TEST(MathOpTest, Atanh) {
  OpTester test("Atanh", 9);
  TrigFloatTest<std::atanh>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Expand_8_3x3) {
  OpTester test("Expand", 8);
  test.AddInput<float>("data_0", {1}, {1.0f});
  test.AddInput<int64_t>("data_1", {2}, {3, 3});
  test.AddOutput<float>("result", {3, 3},
                        {1.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x1) {
  OpTester test("Expand", 8);
  test.AddInput<float>("data_0", {3}, {1.0f, 2.0f, 3.0f});
  test.AddInput<int64_t>("data_1", {2}, {3, 1});
  test.AddOutput<float>("result", {3, 3},
                        {1.0f, 2.0f, 3.0f,
                         1.0f, 2.0f, 3.0f,
                         1.0f, 2.0f, 3.0f});
  test.Run();
}

TEST(MathOpTest, Expand_8_1x3) {
  OpTester test("Expand", 8);
  test.AddInput<float>("data_0", {3, 1}, {1.0f, 2.0f, 3.0f});
  test.AddInput<int64_t>("data_1", {2}, {1, 3});
  test.AddOutput<float>("result", {3, 3},
                        {1.0f, 1.0f, 1.0f,
                         2.0f, 2.0f, 2.0f,
                         3.0f, 3.0f, 3.0f});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x3_int32) {
  OpTester test("Expand", 8);
  test.AddInput<int32_t>("data_0", {1}, {1});
  test.AddInput<int64_t>("data_1", {2}, {3, 3});
  test.AddOutput<int32_t>("result", {3, 3},
                          {1, 1, 1,
                           1, 1, 1,
                           1, 1, 1});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x1_int32) {
  OpTester test("Expand", 8);
  test.AddInput<int32_t>("data_0", {3}, {1, 2, 3});
  test.AddInput<int64_t>("data_1", {2}, {3, 1});
  test.AddOutput<int32_t>("result", {3, 3},
                          {1, 2, 3,
                           1, 2, 3,
                           1, 2, 3});
  test.Run();
}

TEST(MathOpTest, Expand_8_1x3_int32) {
  OpTester test("Expand", 8);
  test.AddInput<int32_t>("data_0", {3, 1}, {1, 2, 3});
  test.AddInput<int64_t>("data_1", {2}, {1, 3});
  test.AddOutput<int32_t>("result", {3, 3},
                          {1, 1, 1,
                           2, 2, 2,
                           3, 3, 3});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x3_int64) {
  OpTester test("Expand", 8);
  test.AddInput<int64_t>("data_0", {1}, {1});
  test.AddInput<int64_t>("data_1", {2}, {3, 3});
  test.AddOutput<int64_t>("result", {3, 3},
                          {1, 1, 1,
                           1, 1, 1,
                           1, 1, 1});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x1_int64) {
  OpTester test("Expand", 8);
  test.AddInput<int64_t>("data_0", {3}, {1, 2, 3});
  test.AddInput<int64_t>("data_1", {2}, {3, 1});
  test.AddOutput<int64_t>("result", {3, 3},
                          {1, 2, 3,
                           1, 2, 3,
                           1, 2, 3});
  test.Run();
}

TEST(MathOpTest, Expand_8_1x3_int64) {
  OpTester test("Expand", 8);
  test.AddInput<int64_t>("data_0", {3, 1}, {1, 2, 3});
  test.AddInput<int64_t>("data_1", {2}, {1, 3});
  test.AddOutput<int64_t>("result", {3, 3},
                          {1, 1, 1,
                           2, 2, 2,
                           3, 3, 3});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x1x3x1_int64) {
  OpTester test("Expand", 8);
  test.AddInput<int64_t>("data_0", {1, 3, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddInput<int64_t>("data_1", {4}, {3, 1, 3, 1});
  test.AddOutput<int64_t>("result", {3, 3, 3, 3},
                          {1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9,
                           1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9,
                           1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x3_float16) {
  OpTester test("Expand", 8);
  test.AddInput<MLFloat16>("data_0", {1}, {MLFloat16(math::floatToHalf(1.0f))});
  test.AddInput<int64_t>("data_1", {2}, {3, 3});
  test.AddOutput<MLFloat16>("result", {3, 3},
                            {MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f)),
                             MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f)),
                             MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f))});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x1_float16) {
  OpTester test("Expand", 8);
  test.AddInput<MLFloat16>("data_0", {3}, {MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(3.0f))});
  test.AddInput<int64_t>("data_1", {2}, {3, 1});
  test.AddOutput<MLFloat16>("result", {3, 3},
                            {MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(3.0f)),
                             MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(3.0f)),
                             MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(3.0f))});
  test.Run();
}

TEST(MathOpTest, Expand_8_1x3_float16) {
  OpTester test("Expand", 8);
  test.AddInput<MLFloat16>("data_0", {3, 1}, {MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(3.0f))});
  test.AddInput<int64_t>("data_1", {2}, {1, 3});
  test.AddOutput<MLFloat16>("result", {3, 3},
                            {MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f)),
                             MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(2.0f)),
                             MLFloat16(math::floatToHalf(3.0f)), MLFloat16(math::floatToHalf(3.0f)), MLFloat16(math::floatToHalf(3.0f))});
  test.Run();
}

TEST(MathOpTest, Erf) {
  OpTester test("Erf", 9);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("A", dims, {0.5f, 1.0f, 0.7f, 2.0f});
  test.AddOutput<float>("B", dims, {0.5204999f, 0.8427008f, 0.6778012f, 0.9953223f});
  test.Run();
}

TEST(MathOpTest, ErfMoreData) {
  OpTester test("Erf", 9);
  std::vector<float> inputs{
      -3.625f, 3.375f, 0.0f, 0.00025f, 0.0005f, -0.00075f, -0.001f, 0.00125f,
      0.0015f, -3.125f, 0.00175f, 2.875f, 2.625f, 2.375f, 2.125f, 6.25e-05f,
      0.0003125f, 0.0005625f, -0.0008125f, 0.0010625f, 0.0013125f, 0.0015625f, 0.0018125f, 3.5625f,
      3.3125f, 3.0625f, 2.8125f, -2.5625f, 2.3125f, 2.0625f, 0.000125f, 0.000375f,
      -0.000625f, -0.000875f, -0.001125f, -0.001375f, -0.001625f, -0.001875f, -3.5f, -3.25f,
      3.0f, 2.75f, -2.5f, -2.25f, -2.0f, -0.0001875f, 0.0004375f, 0.0006875f,
      2.1875f, -1.9375f, 0.0014375f, -0.0016875f, -0.0019375f, 3.4375f, 3.1875f, -2.9375f,
      -2.4375f, -0.0009375f, 0.0011875f};
  std::vector<float> outputs{
      -1.0f, 0.999998f, 0.0f, 0.000282095f, 0.00056419f, -0.000846284f, -0.00112838f, 0.00141047f,
      0.00169257f, -0.99999f, 0.00197466f, 0.999952f, 0.999795f, 0.999217f, 0.997346f, 7.05237e-05f,
      0.000352618f, 0.000634713f, -0.000916808f, 0.0011989f, 0.001481f, 0.00176309f, 0.00204518f, 1.0f,
      0.999997f, 0.999985f, 0.99993f, -0.99971f, 0.998926f, 0.996464f, 0.000141047f, 0.000423142f,
      -0.000705237f, -0.000987331f, -0.00126943f, -0.00155152f, -0.00183361f, -0.00211571f, -0.999999f, -0.999996f,
      0.999978f, 0.999899f, -0.999593f, -0.998537f, -0.995322f, -0.000211571f, 0.000493666f, 0.000775761f,
      0.998022f, -0.993857f, 0.00162204f, -0.00190414f, -0.00218623f, 0.999999f, 0.999993f, -0.999967f,
      -0.999433f, -0.00105786f, 0.00133995f};
  std::vector<int64_t> dims{static_cast<int64_t>(inputs.size())};

  test.AddInput<float>("A", dims, inputs);
  test.AddOutput<float>("B", dims, outputs);
  test.Run();
}

const int ModOp_ver = 10;

TEST(ModOpTest, Fmod_float_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);
  test.AddInput<float>("X", {6}, {-4.3f, 7.2f, 5.0f, 4.3f, -7.2f, 8.0f});
  test.AddInput<float>("Y", {6}, {2.1f, -3.4f, 8.0f, -2.1f, 3.4f, 5.0f});
  test.AddOutput<float>("Z", {6}, {-0.1f, 0.4f, 5.f, 0.1f, -0.4f, 3.f});

  test.Run();
}

std::vector<MLFloat16> MakeMLFloat16(const std::initializer_list<float>& input) {
  std::vector<MLFloat16> output;
  std::transform(input.begin(), input.end(), std::back_inserter(output),
                 [](float fl) {
                   return MLFloat16(math::floatToHalf(fl));
                 });
  return output;
}

TEST(ModOpTest, Fmod_float16_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);

  test.AddInput<MLFloat16>("X", {6}, MakeMLFloat16({-4.3f, 7.2f, 5.0f, 4.3f, -7.2f, 8.0f}));
  test.AddInput<MLFloat16>("Y", {6}, MakeMLFloat16({2.1f, -3.4f, 8.0f, -2.1f, 3.4f, 5.0f}));
  // The output above is {-0.1f, 0.4f, 5.f, 0.1f, -0.4f, 3.f} for float
  test.AddOutput<MLFloat16>("Z", {6}, MakeMLFloat16({-0.1015625f, 0.3984375f, 5.f, 0.1015625f, -0.3984375f, 3.f}));

  test.Run();
}

TEST(ModOpTest, Int8_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int8_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int8_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int8_t>("Z", {6}, {0, -2, 5, 0, 2, 3});

  test.Run();
}

TEST(ModOpTest, Int8_mixed_sign_fmod) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);

  test.AddInput<int8_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int8_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int8_t>("Z", {6}, {0, 1, 5, 0, -1, 3});

  test.Run();
}

TEST(ModOpTest, UInt8_mod) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<uint8_t>("X", {6}, {4, 7, 5, 4, 7, 8});
  test.AddInput<uint8_t>("Y", {6}, {2, 3, 8, 2, 3, 5});
  test.AddOutput<uint8_t>("Z", {6}, {0, 1, 5, 0, 1, 3});

  test.Run();
}

TEST(ModOpTest, Int16_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int16_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int16_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int16_t>("Z", {6}, {0, -2, 5, 0, 2, 3});

  test.Run();
}

TEST(ModOpTest, Int16_mixed_sign_fmod) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);

  test.AddInput<int16_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int16_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int16_t>("Z", {6}, {0, 1, 5, 0, -1, 3});

  test.Run();
}

TEST(ModOpTest, UInt16_mod) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<uint16_t>("X", {6}, {4, 7, 5, 4, 7, 8});
  test.AddInput<uint16_t>("Y", {6}, {2, 3, 8, 2, 3, 5});
  test.AddOutput<uint16_t>("Z", {6}, {0, 1, 5, 0, 1, 3});

  test.Run();
}

TEST(ModOpTest, Int32_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int32_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int32_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int32_t>("Z", {6}, {0, -2, 5, 0, 2, 3});

  test.Run();
}

TEST(ModOpTest, Int32_mixed_sign_fmod) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);

  test.AddInput<int32_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int32_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int32_t>("Z", {6}, {0, 1, 5, 0, -1, 3});

  test.Run();
}

TEST(ModOpTest, UInt32_mod) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<uint32_t>("X", {6}, {4, 7, 5, 4, 7, 8});
  test.AddInput<uint32_t>("Y", {6}, {2, 3, 8, 2, 3, 5});
  test.AddOutput<uint32_t>("Z", {6}, {0, 1, 5, 0, 1, 3});

  test.Run();
}

TEST(ModOpTest, Int64_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int64_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int64_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int64_t>("Z", {6}, {0, -2, 5, 0, 2, 3});

  test.Run();
}

TEST(ModOpTest, Int64_mixed_sign_fmod) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);

  test.AddInput<int64_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int64_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int64_t>("Z", {6}, {0, 1, 5, 0, -1, 3});

  test.Run();
}

TEST(ModOpTest, UInt64_mod) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<uint64_t>("X", {6}, {4, 7, 5, 4, 7, 8});
  test.AddInput<uint64_t>("Y", {6}, {2, 3, 8, 2, 3, 5});
  test.AddOutput<uint64_t>("Z", {6}, {0, 1, 5, 0, 1, 3});

  test.Run();
}

TEST(ModOpTest, Int32_mod_bcast) {
  OpTester test("Mod", ModOp_ver);

  std::vector<int32_t> input_sequence;
  input_sequence.resize(30);
  std::generate(input_sequence.begin(), input_sequence.end(),
                [n = 0]() mutable { return n++; });

  // input [0..29]
  test.AddInput<int32_t>("X", {3, 2, 5}, input_sequence);
  test.AddInput<int32_t>("Y", {1}, {7});

  test.AddOutput<int32_t>("Z", {3, 2, 5},
                          {0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
