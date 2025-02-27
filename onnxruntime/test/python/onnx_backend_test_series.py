# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import sys
import os
import platform
import unittest
import onnx
import onnx.backend.test

import numpy as np
import onnxruntime.backend as c2

pytest_plugins = 'onnx.backend.test.report',

class OrtBackendTest(onnx.backend.test.BackendTest):

    def __init__(self, backend, parent_module=None):
        super(OrtBackendTest, self).__init__(backend, parent_module)

    @classmethod
    def assert_similar_outputs(cls, ref_outputs, outputs, rtol, atol):
        np.testing.assert_equal(len(ref_outputs), len(outputs))
        for i in range(len(outputs)):
            np.testing.assert_equal(ref_outputs[i].dtype, outputs[i].dtype)
            if ref_outputs[i].dtype == np.object:
                np.testing.assert_array_equal(ref_outputs[i], outputs[i])
            else:
                np.testing.assert_allclose(
                    ref_outputs[i],
                    outputs[i],
                    rtol=1e-3,
                    atol=1e-5)


# ORT first supported opset 7, so models with nodes that require versions prior to opset 7 are not supported
def tests_with_pre_opset7_dependencies_filters():
    filters = ('^test_AvgPool1d_cpu.*',
               '^test_AvgPool1d_stride_cpu.*',
               '^test_AvgPool2d_cpu.*',
               '^test_AvgPool2d_stride_cpu.*',
               '^test_AvgPool3d_cpu.*',
               '^test_AvgPool3d_stride1_pad0_gpu_input_cpu.*',
               '^test_AvgPool3d_stride_cpu.*',
               '^test_BatchNorm1d_3d_input_eval_cpu.*',
               '^test_BatchNorm2d_eval_cpu.*',
               '^test_BatchNorm2d_momentum_eval_cpu.*',
               '^test_BatchNorm3d_eval_cpu.*',
               '^test_BatchNorm3d_momentum_eval_cpu.*',
               '^test_GLU_cpu.*',
               '^test_GLU_dim_cpu.*',
               '^test_Linear_cpu.*',
               '^test_PReLU_1d_cpu.*',
               '^test_PReLU_1d_multiparam_cpu.*',
               '^test_PReLU_2d_cpu.*',
               '^test_PReLU_2d_multiparam_cpu.*',
               '^test_PReLU_3d_cpu.*',
               '^test_PReLU_3d_multiparam_cpu.*',
               '^test_PoissonNLLLLoss_no_reduce_cpu.*',
               '^test_Softsign_cpu.*',
               '^test_operator_add_broadcast_cpu.*',
               '^test_operator_add_size1_broadcast_cpu.*',
               '^test_operator_add_size1_right_broadcast_cpu.*',
               '^test_operator_add_size1_singleton_broadcast_cpu.*',
               '^test_operator_addconstant_cpu.*',
               '^test_operator_addmm_cpu.*',
               '^test_operator_basic_cpu.*',
               '^test_operator_mm_cpu.*',
               '^test_operator_non_float_params_cpu.*',
               '^test_operator_params_cpu.*',
               '^test_operator_pow_cpu.*')

    return filters


def unsupported_usages_filters():
    filters = ('^test_convtranspose_1d_cpu.*',  # ConvTransponse supports 4-D only
               '^test_convtranspose_3d_cpu.*')

    return filters


def create_backend_test(testname=None):
    backend_test = OrtBackendTest(c2, __name__)

    # Type not supported
    backend_test.exclude(r'(FLOAT16)')

    if testname:
        backend_test.include(testname + '.*')
    else:
        # Tests that are failing temporarily and should be fixed
        current_failing_tests = ('^test_cast_STRING_to_FLOAT_cpu.*',
                                 '^test_cast_FLOAT_to_STRING_cpu.*',
                                 '^test_qlinearconv_cpu.*',
                                 '^test_gru_seq_length_cpu.*',
                                 '^test_bitshift_right_uint16_cpu.*',
                                 '^test_bitshift_right_uint32_cpu.*',
                                 '^test_bitshift_right_uint64_cpu.*',
                                 '^test_bitshift_right_uint8_cpu.*',
                                 '^test_bitshift_left_uint16_cpu.*',
                                 '^test_bitshift_left_uint32_cpu.*',
                                 '^test_bitshift_left_uint64_cpu.*',
                                 '^test_bitshift_left_uint8_cpu.*',
                                 '^test_round_cpu.*',
                                 '^test_cumsum_1d_cpu.*',
                                 '^test_cumsum_1d_exclusive_cpu.*',
                                 '^test_cumsum_1d_reverse_cpu.*',
                                 '^test_cumsum_1d_reverse_exclusive_cpu.*',
                                 '^test_cumsum_2d_axis_0_cpu.*',
                                 '^test_cumsum_2d_axis_1_cpu.*',
                                 '^test_dynamicquantizelinear_expanded*',
                                 '^test_dynamicquantizelinear_max_adjusted_expanded*',
                                 '^test_dynamicquantizelinear_min_adjusted_expanded*',
                                 '^test_gather_elements*',
                                 '^test_scatter_elements*',
                                 '^test_top_k*',
                                 '^test_unique_*',
                                 '^test_mod_float_mixed_sign_example_cpu.*', #onnxruntime::Mod::Compute fmod_ was false. fmod attribute must be true for float, float16 and double types
                                 '^test_shrink_cpu.*', #Invalid rank for input: x Got: 1 Expected: 2 Please fix either the inputs or the model.
                                 '^test_range_float_type_positive_delta_cpu.*',
                                 '^test_range_float_type_positive_delta_expanded_cpu.*',
                                 '^test_range_int32_type_negative_delta_cpu.*',
                                 '^test_range_int32_type_negative_delta_expanded_cpu.*',
                                 '^test_det_2d_cpu.*',
                                 '^test_det_nd_cpu.*',
                                 '^test_gathernd_example_float32_cpu.*',
                                 '^test_gathernd_example_int32_cpu.*',
                                 '^test_resize_downsample_scales_cubic_A_n0p5_exclude_outside_cpu.*',
                                 '^test_resize_downsample_scales_cubic_align_corners_cpu.*',
                                 '^test_resize_downsample_scales_cubic_cpu.*',
                                 '^test_resize_downsample_scales_linear_align_corners_cpu.*',
                                 '^test_resize_downsample_scales_linear_cpu.*',
                                 '^test_resize_downsample_scales_nearest_cpu.*',
                                 '^test_resize_downsample_sizes_cubic_cpu.*',
                                 '^test_resize_downsample_sizes_linear_pytorch_half_pixel_cpu.*',
                                 '^test_resize_downsample_sizes_nearest_cpu.*',
                                 '^test_resize_downsample_sizes_nearest_tf_half_pixel_for_nn_cpu.*',
                                 '^test_resize_tf_crop_and_resize_cpu.*',
                                 '^test_resize_upsample_scales_cubic_A_n0p5_exclude_outside_cpu.*',
                                 '^test_resize_upsample_scales_cubic_align_corners_cpu.*',
                                 '^test_resize_upsample_scales_cubic_asymmetric_cpu.*',
                                 '^test_resize_upsample_scales_cubic_cpu.*',
                                 '^test_resize_upsample_scales_linear_align_corners_cpu.*',
                                 '^test_resize_upsample_scales_linear_cpu.*',
                                 '^test_resize_upsample_scales_nearest_cpu.*',
                                 '^test_resize_upsample_sizes_cubic_cpu.*',
                                 '^test_resize_upsample_sizes_nearest_ceil_half_pixel_cpu.*',
                                 '^test_resize_upsample_sizes_nearest_cpu.*',
                                 '^test_resize_upsample_sizes_nearest_floor_align_corners_cpu.*',
                                 '^test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric_cpu.*',
                                 '^test_scatternd_cpu.*',
                                 )

        # Example of how to disable tests for a specific provider.
        # if c2.supports_device('NGRAPH'):
        #    current_failing_tests = current_failing_tests + ('|^test_operator_repeat_dim_overflow_cpu.*',)
        if c2.supports_device('NGRAPH'):
            current_failing_tests = current_failing_tests + ('|^test_clip*',)
            current_failing_tests = current_failing_tests + ('|^test_depthtospace_crd*',)

        if c2.supports_device('OPENVINO_GPU_FP32') or c2.supports_device('OPENVINO_GPU_FP16'):
            current_failing_tests = current_failing_tests + ('^test_div_cpu*',)

        filters = current_failing_tests + \
                  tests_with_pre_opset7_dependencies_filters() + \
                  unsupported_usages_filters()

        backend_test.exclude('(' + '|'.join(filters) + ')')
        print ('excluded tests:', filters)

    # import all test cases at global scope to make
    # them visible to python.unittest.
    globals().update(backend_test.enable_report().test_cases)

    return backend_test


def parse_args():
    parser = argparse.ArgumentParser(os.path.basename(__file__),
                                     description='Run the ONNX backend tests using ONNXRuntime.')

    # Add an argument to match a single test name, by adding the name to the 'include' filter.
    # Using -k with python unittest (https://docs.python.org/3/library/unittest.html#command-line-options)
    # doesn't work as it filters on the test method name (Runner._add_model_test) rather than inidividual test case names.
    parser.add_argument('-t', '--test-name', dest='testname', type=str,
                        help="Only run tests that match this value. Matching is regex based, and '.*' is automatically appended")

    # parse just our args. python unittest has its own args and arg parsing, and that runs inside unittest.main()
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    return args


if __name__ == '__main__':
    args = parse_args()

    backend_test = create_backend_test(args.testname)
    unittest.main()
