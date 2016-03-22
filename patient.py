# -*- coding: utf-8 -*-
from __future__ import division, print_function  # python 2/3 interoperability
import numpy as np
from datetime import datetime
import time
import sys

# Ensure that Python 3.5 is used
assert sys.version_info >= (3, 5)


class GPUTiming:
    def __init__(self, name=None):
        """
        Simple class to make it easier to display timing for each kernel
        :param name: Name of the gpu function/kernel/etc...
        """

        self.host_to_device = 0.0
        self.device_to_host = 0.0
        self.kernel_run_time = 0.0
        self.init_on_device = 0.0
        self.kernel_name = name
        self.total_time = 0.0

    def print_timings(self):
        print("GPU Function:", self.kernel_name)
        print("Init on Device: \t\t\t{0:.5f} seconds".format(
            self.init_on_device))
        print("Host to Device: \t\t\t{0:.5f} seconds".format(
            self.host_to_device))
        print("Kernel Runtime: \t\t\t{0:.5f} seconds".format(
            self.kernel_run_time))
        print("Device to Host: \t\t\t{0:.5f} seconds".format(
            self.device_to_host))
        print("\t\t\t\t\t---------------")
        print("Total: \t\t\t\t\t{0:.5f} seconds".format(
            self.init_on_device +
            self.device_to_host +
            self.host_to_device +
            self.kernel_run_time))
        print()
        print()


class Lead:
    def __init__(self,
                 lead_number=None,
                 voltage=None,
                 samples=None,
                 resolution=None,
                 sample_rate=None,
                 time_series=None):
        """

        :param lead_number:
        :param voltage: Voltage signal data
        :param samples:
        :param resolution:
        :param sample_rate: Sampling rate of the lead
        :param time_series: Time corresponding to signal data
        :return:
        """

        self.number = lead_number
        self.voltage_data = voltage
        self.samples = samples
        self.resolution = resolution
        self.sample_rate = sample_rate
        self.heart_rate = None
        self.rr_peaks = None
        self.voltage_first_derivative = None
        self.voltage_second_derivative = None

        # Array the holds the time increments as numpy datetime format
        self.time = time_series

        self.heart_rate_computed = False

        # Index of the first r peak
        self.first_r_peak_index = None

    def get_first_r_peak_index(self):
        """
        Find the first r peak in the signal. This is used to compute offset
        values when comparing different leads
        """

        self.first_r_peak_index = 0

    def compute_first_deriv_cpu(self, lead_number=None):
        """
        Compute the derivatives of a given lead signal
        :param lead_number: Intger of the lead number to calculate the derivative of
        :return: Returns an array of the derivatives from the specified lead number
        """

        print("Computing derivative for lead", lead_number, "on the cpu...")

        voltage_data = (self.voltage_data *
                       (self.resolution / 1000000.0)).astype(np.float32)

        dx = self.samples / self.sample_rate / self.samples

        self.voltage_first_derivative = np.diff(voltage_data) / dx

    def compute_first_deriv_gpu(self, transfer_to_host=False):
        """
        Send the raw lead voltages to the gpu and calculate the 1st derivative
        :param transfer_to_host: True/False to transfer dfdx to the cpu
        :return: gpuarray of derivatives (still on gpu), and GPUTiming object
        """

        # Try to import the pyCUDA module
        try:
            from pycuda import driver, compiler, gpuarray, tools, cumath
            import pycuda.autoinit
            from pycuda.compiler import SourceModule

        except ImportError:
            print("Error: Unable to import the pyCUDA package")
            return

        module = SourceModule(
            """
            // Simple gpu kernel to compute the derivative
            // with a constant dx (using shared memory)
            __global__ void dfdx_shared(float* f,
                                        float* f_shifted,
                                        float* dfdx,
                                        float dx,
                                        int n)
            {
                unsigned int g_i = blockIdx.x * blockDim.x + threadIdx.x;
                unsigned int s_i = threadIdx.x;

                __shared__ float shared_dx;
                __shared__ float shared_f[512];
                __shared__ float shared_f_shifted[512];

                // Assigned to shared memory
                shared_dx = dx;
                shared_f[s_i] = f[g_i];
                shared_f_shifted[s_i] = f_shifted[g_i];

                __syncthreads();

                dfdx[g_i] = (shared_f_shifted[s_i] - shared_f[s_i]) / shared_dx;
            }
            """)

        voltage_data = (self.voltage_data *
                       (self.resolution / 1000000.0)).astype(np.float32)

        dx = self.samples / self.sample_rate / self.samples
        n = self.voltage_data[2:].size
        alpha = np.float32(1.0 / dx ** 2)

        start_t_gpu = driver.Event()
        end_t_gpu = driver.Event()

        # Create a timing object
        timing = GPUTiming(name='1st Derivative Kernel')

        # GPU Arrays
        start_t_gpu.record()

        voltage_gpu = gpuarray.to_gpu(voltage_data[:-2])
        voltage_gpu_shifted = gpuarray.to_gpu(voltage_data[1:-1])

        end_t_gpu.record()
        end_t_gpu.synchronize()

        timing.host_to_device = start_t_gpu.time_till(end_t_gpu) * 1e-3

        # Initialize an empty array on the gpu to hold the derivatives
        start_t_gpu.record()
        dfdx_gpu = gpuarray.zeros(n, np.float32)
        end_t_gpu.record()
        end_t_gpu.synchronize()

        timing.init_on_device = start_t_gpu.time_till(end_t_gpu) * 1e-3

        # Device info
        # gpu_dev = tools.DeviceData()
        threads_per_block = 512  # gpu_dev.max_threads
        n_blocks = int(np.ceil(n / threads_per_block))

        # Run the kernel
        derivative_kernel = module.get_function("dfdx_shared")
        start_t_gpu.record()
        derivative_kernel(voltage_gpu,
                          voltage_gpu_shifted,
                          dfdx_gpu,
                          alpha,
                          np.int_(n),
                          block=(threads_per_block, 1, 1),
                          grid=(n_blocks, 1, 1))

        end_t_gpu.record()
        end_t_gpu.synchronize()
        timing.kernel_run_time = start_t_gpu.time_till(end_t_gpu)*1e-3

        if transfer_to_host:
            start_t_gpu.record()
            dfdx_cpu = dfdx_gpu.get()
            end_t_gpu.record()
            end_t_gpu.synchronize()
            timing.device_to_host = start_t_gpu.time_till(end_t_gpu)*1e-3
            return dfdx_cpu, timing

        else:
            return dfdx_gpu, timing

    def compute_second_deriv_gpu(self, transfer_to_host=False):
        """
        Send the raw lead voltages to the gpu and calculate the 2nd derivative
        :param transfer_to_host: True/False to transfer dfdx to the cpu
        :return: gpuarray of derivatives (still on gpu), and GPUTiming object
        """

        # Try to import the pyCUDA module
        try:
            from pycuda import driver, compiler, gpuarray, tools, cumath
            import pycuda.autoinit
            from pycuda.compiler import SourceModule

        except ImportError:
            print("Error: Unable to import the pyCUDA package")
            return

        module = SourceModule(
            """
            // Simple gpu kernel to compute the 2nd derivative
            // with a constant dx (using shared memory)

            __global__ void d2fdx2(float* f,
                                          float* f_s,
                                          float* f_ss,
                                          float* d2fdx2,
                                          float alpha,
                                          int n)
            {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

                if (i > n){
                    return;
                }

                d2fdx2[i] = alpha * (f[i] - 2 * f_s[i] + f_ss[i]);
            }

            __global__ void d2fdx2_shared(float* f,
                                          float* f_s,
                                          float* f_ss,
                                          float* d2fdx2,
                                          float alpha,
                                          int n)
            {
                unsigned int g_i = blockIdx.x * blockDim.x + threadIdx.x;
                unsigned int s_i = threadIdx.x;

                if (g_i > n){
                    return;
                }
                __shared__ float shared_alpha;
                __shared__ float shared_f[512];
                __shared__ float shared_f_s[512];
                __shared__ float shared_f_ss[512];

                // Assigned to shared memory
                shared_alpha = alpha;
                shared_f[s_i] = f[g_i];
                shared_f_s[s_i] = f_s[g_i];
                shared_f_ss[s_i] = f_ss[g_i];

                __syncthreads();

                d2fdx2[g_i] = shared_alpha * (shared_f[s_i] - 2 * shared_f_s[s_i] + shared_f_ss[s_i]);
            }
            """)

        voltage_data = (self.voltage_data *
                       (self.resolution / 1000000.0)).astype(np.float32)

        dx = self.samples / self.sample_rate / self.samples
        n = self.voltage_data[2:].size
        alpha = np.float32(1.0 / dx ** 2)

        start_t_gpu = driver.Event()
        end_t_gpu = driver.Event()

        # Create a timing object
        timing = GPUTiming(name='2nd Derivative Kernel')

        # GPU Arrays
        start_t_gpu.record()

        voltage_gpu = gpuarray.to_gpu(voltage_data[:-2])
        voltage_gpu_shifted = gpuarray.to_gpu(voltage_data[1:-1])
        voltage_gpu_shifted_again = gpuarray.to_gpu(voltage_data[2:])

        end_t_gpu.record()
        end_t_gpu.synchronize()

        timing.host_to_device = start_t_gpu.time_till(end_t_gpu) * 1e-3

        # Initialize an empty array on the gpu to hold the derivatives
        start_t_gpu.record()
        df2dx2_gpu = gpuarray.empty(n, np.float32)
        end_t_gpu.record()
        end_t_gpu.synchronize()

        timing.init_on_device = start_t_gpu.time_till(end_t_gpu) * 1e-3

        # Device info
        # gpu_dev = tools.DeviceData()
        threads_per_block = 512  # gpu_dev.max_threads
        n_blocks = int(np.ceil(n / threads_per_block))

        # Run the kernel
        derivative_kernel = module.get_function("d2fdx2_shared")
        start_t_gpu.record()
        derivative_kernel(voltage_gpu,
                          voltage_gpu_shifted,
                          voltage_gpu_shifted_again,
                          df2dx2_gpu,
                          alpha,
                          np.int_(n),
                          block=(threads_per_block, 1, 1),
                          grid=(n_blocks, 1, 1))

        end_t_gpu.record()
        end_t_gpu.synchronize()
        timing.kernel_run_time = start_t_gpu.time_till(end_t_gpu)*1e-3

        if transfer_to_host:
            start_t_gpu.record()
            df2dx2_cpu = df2dx2_gpu.get()
            end_t_gpu.record()
            end_t_gpu.synchronize()
            timing.device_to_host = start_t_gpu.time_till(end_t_gpu)*1e-3
            return df2dx2_cpu, timing

        else:
            return df2dx2_gpu, timing

    def compute_second_deriv_gpu_non_shift(self, transfer_to_host=False):
        """
        Send the raw lead voltages to the gpu and calculate the 2nd derivative
        :param transfer_to_host: True/False to transfer dfdx to the cpu
        :return: gpuarray of derivatives (still on gpu), and GPUTiming object
        """

        # Try to import the pyCUDA module
        try:
            from pycuda import driver, compiler, gpuarray, tools, cumath
            import pycuda.autoinit
            from pycuda.compiler import SourceModule

        except ImportError:
            print("Error: Unable to import the pyCUDA package")
            return

        module = SourceModule(
            """
            // Simple gpu kernel to compute the 2nd derivative
            // with a constant dx (using shared memory)
            __global__ void d2fdx2(float* f,
                                   float* d2fdx2,
                                   float alpha,
                                   int n)
            {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                if ((i > 1) && (i < n))
                {
                    d2fdx2[i] = alpha * (f[i+1] - 2 * f[i] + f[i-1]);
                }
            }
            """)

        voltage_data = (self.voltage_data *
                       (self.resolution / 1000000.0)).astype(np.float32)

        dx = self.samples / self.sample_rate / self.samples
        n = self.voltage_data.size
        alpha = np.float32(1.0 / dx ** 2)

        start_t_gpu = driver.Event()
        end_t_gpu = driver.Event()

        # Create a timing object
        timing = GPUTiming(name='2nd Derivative Kernel - Non Shifted')

        # GPU Arrays
        start_t_gpu.record()

        voltage_gpu = gpuarray.to_gpu(voltage_data)

        end_t_gpu.record()
        end_t_gpu.synchronize()

        timing.host_to_device = start_t_gpu.time_till(end_t_gpu) * 1e-3

        # Initialize an empty array on the gpu to hold the derivatives
        start_t_gpu.record()
        df2dx2_gpu = gpuarray.empty(n, np.float32)
        end_t_gpu.record()
        end_t_gpu.synchronize()

        timing.init_on_device = start_t_gpu.time_till(end_t_gpu) * 1e-3

        # Device info
        # gpu_dev = tools.DeviceData()
        threads_per_block = 512  # gpu_dev.max_threads
        n_blocks = int(np.ceil(n / threads_per_block))

        # Run the kernel
        derivative_kernel = module.get_function("d2fdx2")
        start_t_gpu.record()
        derivative_kernel(voltage_gpu,
                          df2dx2_gpu,
                          alpha,
                          np.int_(n),
                          block=(threads_per_block, 1, 1),
                          grid=(n_blocks, 1, 1))

        end_t_gpu.record()
        end_t_gpu.synchronize()
        timing.kernel_run_time = start_t_gpu.time_till(end_t_gpu)*1e-3

        if transfer_to_host:
            start_t_gpu.record()
            df2dx2_cpu = df2dx2_gpu.get()
            end_t_gpu.record()
            end_t_gpu.synchronize()
            timing.device_to_host = start_t_gpu.time_till(end_t_gpu)*1e-3
            return df2dx2_cpu, timing

        else:
            return df2dx2_gpu, timing

    def threshold_above_gpu(self,
                            threshold_value=None,
                            gpu_array=None,
                            transfer_to_host=True):
        """

        :param threshold_value:
        :param gpu_array: Array with values to threshold
        :param transfer_to_host: True/False to transfer thresholded to the cpu
        :return:
        """

        # Try to import the pyCUDA module
        try:
            from pycuda import driver, compiler, gpuarray, tools, cumath
            import pycuda.autoinit
            from pycuda.compiler import SourceModule

        except ImportError:
            print("Error: Unable to import the pyCUDA package")
            return

        module = SourceModule(
            """
            // Filter out values that are not big enough
            // (use for very large derivatives) i.e. large positive spikes
            __global__ void get_only_large_deriv_shared(float* f,
                                                        int* thresholded_f,
                                                        float threshold)
            {
                __shared__ float shared_f[512];
                __shared__ int  shared_thresholded_f[512];
                //float shared_threshold;

                unsigned int g_i = blockIdx.x * blockDim.x + threadIdx.x;
                unsigned int b_i = threadIdx.x;

                //shared_threshold = threshold;

                // Put f into shared memory
                shared_f[b_i] = f[g_i];

                __syncthreads();

                // 1 if at or above threshold, 0 if not
                shared_thresholded_f[b_i] = (shared_f[b_i] >= threshold) ? 1 : 0;

                // Put threshold values into global memory
                thresholded_f[g_i] = shared_thresholded_f[b_i];

            }
            """)

        n = self.voltage_data[2:].size

        start_t_gpu = driver.Event()
        end_t_gpu = driver.Event()

        # Create a timing object
        timing = GPUTiming(name='Threshold Above Kernel')

        # Initialize an empty array on the gpu to hold the derivatives
        start_t_gpu.record()
        thresholded_array = gpuarray.empty(n, np.int32)
        end_t_gpu.record()
        end_t_gpu.synchronize()

        timing.init_on_device = start_t_gpu.time_till(end_t_gpu) * 1e-3

        # Device info
        # gpu_dev = tools.DeviceData()
        threads_per_block = 512  # gpu_dev.max_threads
        n_blocks = int(np.ceil(n / threads_per_block))

        # Run the kernel
        threshold_kernel = module.get_function("get_only_large_deriv_shared")
        start_t_gpu.record()
        threshold_kernel(gpu_array,
                         thresholded_array,
                         np.float32(threshold_value),
                         block=(threads_per_block, 1, 1),
                         grid=(n_blocks, 1, 1))

        end_t_gpu.record()
        end_t_gpu.synchronize()
        timing.kernel_run_time = start_t_gpu.time_till(end_t_gpu)*1e-3

        if transfer_to_host:
            start_t_gpu.record()
            thresholded_array_cpu = thresholded_array.get()
            end_t_gpu.record()
            end_t_gpu.synchronize()
            timing.device_to_host = start_t_gpu.time_till(end_t_gpu)*1e-3
            return thresholded_array_cpu, timing

        else:
            return thresholded_array, timing

    def threshold_below_gpu(self,
                            threshold_value=None,
                            gpu_array=None,
                            transfer_to_host=True):
        """

        :param threshold_value:
        :param gpu_array: Array with values to threshold
        :param transfer_to_host: True/False to transfer thresholded to the cpu
        :return:
        """

        # Try to import the pyCUDA module
        try:
            from pycuda import driver, compiler, gpuarray, tools, cumath
            import pycuda.autoinit
            from pycuda.compiler import SourceModule

        except ImportError:
            print("Error: Unable to import the pyCUDA package")
            return

        module = SourceModule(
            """
            // Filter out values that are not big enough
            // (use for very large derivatives) i.e. large positive spikes
            __global__ void get_only_small_deriv_shared(float* f,
                                                        int* thresholded_f,
                                                        float threshold)
            {
                __shared__ float shared_f[512];
                __shared__ int  shared_thresholded_f[512];
                //float shared_threshold;

                unsigned int g_i = blockIdx.x * blockDim.x + threadIdx.x;
                unsigned int b_i = threadIdx.x;

                //shared_threshold = threshold;

                // Put f into shared memory
                shared_f[b_i] = f[g_i];

                __syncthreads();

                // 1 if at or above threshold, 0 if not
                shared_thresholded_f[b_i] = (shared_f[b_i] <= threshold) ? 1 : 0;

                // Put threshold values into global memory
                thresholded_f[g_i] = shared_thresholded_f[b_i];

            }
            """)

        n = self.voltage_data[2:].size

        start_t_gpu = driver.Event()
        end_t_gpu = driver.Event()

        # Create a timing object
        timing = GPUTiming(name='Threshold Below Kernel')

        # Initialize an empty array on the gpu to hold the derivatives
        start_t_gpu.record()
        thresholded_array = gpuarray.zeros(n, np.int32)
        end_t_gpu.record()
        end_t_gpu.synchronize()

        timing.init_on_device = start_t_gpu.time_till(end_t_gpu) * 1e-3

        # Device info
        # gpu_dev = tools.DeviceData()
        threads_per_block = 512  # gpu_dev.max_threads
        n_blocks = np.int_(np.ceil(n / threads_per_block))

        # Run the kernel
        threshold_kernel = module.get_function("get_only_small_deriv_shared")
        start_t_gpu.record()
        threshold_kernel(gpu_array,
                         thresholded_array,
                         threshold_value,
                         block=(threads_per_block, 1, 1),
                         grid=(n_blocks, 1, 1))

        end_t_gpu.record()
        end_t_gpu.synchronize()
        timing.kernel_run_time = start_t_gpu.time_till(end_t_gpu)*1e-3

        if transfer_to_host:
            start_t_gpu.record()
            thresholded_array_cpu = thresholded_array.get()
            end_t_gpu.record()
            end_t_gpu.synchronize()
            timing.device_to_host = start_t_gpu.time_till(end_t_gpu)*1e-3
            return thresholded_array_cpu, timing

        else:
            return thresholded_array, timing

    @staticmethod
    def find_r_peaks(r_binary_array=None):
        """
        Look for the r peaks in the signal
        :param r_binary_array: Array that holds 1's or 0's to show where
        suspected peaks may be
        :return: array with indices of r peaks
        """

        # Find the locations of the filtered derivatives,
        # suspected to be r peaks
        start_t_cpu = time.clock()

        rr_peaks = np.ascontiguousarray(np.where(r_binary_array != 0)[0])

        end_t_cpu = time.clock()
        run_time = end_t_cpu - start_t_cpu
        print("Locate R peaks on CPU: \t\t\t{0:.5f} seconds".format(run_time))

        return rr_peaks, run_time

    def rr_to_hr_gpu(self,
                     rr_peaks=None,
                     transfer_to_host=True):
        """

        :param rr_peaks:
        :param transfer_to_host:
        :return:
        """
        # Try to import the pyCUDA module
        try:
            from pycuda import driver, compiler, gpuarray, tools, cumath
            import pycuda.autoinit
            from pycuda.compiler import SourceModule

        except ImportError:
            print("Error: Unable to import the pyCUDA package")
            return

        module = SourceModule(
            """
            __global__ void rr_peaks(float* r_peaks,
                                     float* r_peaks_shifted,
                                     float* hr,
                                     float alpha)
            {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

                // alpha is 60/dt
                // hr is the distance between peaks div * 2 / by time
                // i.e. 2 beats (distance/time) = beats per time

                hr[i] = alpha / (r_peaks_shifted[i] - r_peaks[i]);
            }
            """)

        start_t_gpu = driver.Event()
        end_t_gpu = driver.Event()

        # Create a timing object
        timing = GPUTiming(name='RR to Heart Rate Kernel')

        # GPU Arrays
        start_t_gpu.record()

        rr_peaks_gpu = gpuarray.to_gpu(rr_peaks[:-1].astype(np.float32))
        rr_peaks_gpu_shifted = gpuarray.to_gpu(rr_peaks[1:].astype(np.float32))

        end_t_gpu.record()
        end_t_gpu.synchronize()

        timing.host_to_device = start_t_gpu.time_till(end_t_gpu) * 1e-3

        # Initialize an empty array on the gpu to hold the derivatives
        start_t_gpu.record()
        hr_gpu = gpuarray.zeros(rr_peaks.size, np.float32)
        end_t_gpu.record()
        end_t_gpu.synchronize()

        timing.init_on_device = start_t_gpu.time_till(end_t_gpu) * 1e-3

        # Device info
        # gpu_dev = tools.DeviceData()
        threads_per_block = 512  # gpu_dev.max_threads
        n_blocks = np.int_(np.ceil(rr_peaks.size / threads_per_block))

        dx = self.samples / self.sample_rate / self.samples
        alpha = np.float32(60.0 / dx)

        # Run the kernel
        rr_peaks_kernel = module.get_function("rr_peaks")
        start_t_gpu.record()
        rr_peaks_kernel(rr_peaks_gpu,
                        rr_peaks_gpu_shifted,
                        hr_gpu,
                        alpha,
                        block=(threads_per_block, 1, 1),
                        grid=(n_blocks, 1, 1))

        end_t_gpu.record()
        end_t_gpu.synchronize()
        timing.kernel_run_time = start_t_gpu.time_till(end_t_gpu)*1e-3

        if transfer_to_host:
            start_t_gpu.record()
            hr_cpu = hr_gpu.get()
            end_t_gpu.record()
            end_t_gpu.synchronize()
            timing.device_to_host = start_t_gpu.time_till(end_t_gpu)*1e-3
            return hr_cpu, timing

        else:
            return hr_gpu, timing

    def threshold_hr_values_cpu(self,
                                heart_rate_data=None,
                                r_peak_array=None,
                                min_heart_rate=30.0,
                                max_heart_rate=200.0):
        """
        Filter out the heart rate values that don't fall within a certain range

        :param heart_rate_data: Array holding all of the heart rate values
        :param r_peak_array: Array holding all of the suspected r peaks
        :param min_heart_rate: Minimum heart rate (Default to 30)
        :param max_heart_rate: Maximum heart rate (Default to 200)
        :return:
        """

        start_t_cpu = time.clock()

        # Find where the heart rate is high enough to be valid
        good_indices = np.where((heart_rate_data >= min_heart_rate) &
                                (heart_rate_data <= max_heart_rate))[0]

        filtered_hr = np.zeros(good_indices.size, dtype='float32')
        filtered_t = np.zeros(good_indices.size, dtype='float32')

        # Loop through the good heart rates
        for i, v in np.ndenumerate(good_indices):

            # Get the filtered heart rate value
            filtered_hr[i] = heart_rate_data[good_indices[i]]

            # Get the corresponding time that the heart rate occurs
            filtered_t[i] = r_peak_array[good_indices[i]]

        end_t_cpu = time.clock()

        run_time = end_t_cpu - start_t_cpu

        # print("Heart Rate Filtering on CPU: \t\t{0:.5f} seconds".format(run_time))

        # Save it to the object -> array[time, heart_rate at time]
        self.heart_rate = [None, None]
        self.heart_rate[1] = np.copy(filtered_hr)
        self.heart_rate[0] = np.copy(filtered_t)

        self.heart_rate_computed = True

        return run_time

    def threshold_hr_values_gpu(self):
        pass

    def get_heart_rate(self):
        """
        Compute the heart rate
        :return:
        """

        d2fdx2_gpuarray, deriv_timing = self.compute_second_deriv_gpu(
            transfer_to_host=False)
        d2fdx2_max = 18000.0
        deriv_timing.print_timings()

        d2fdx2_gpuarray, deriv2_timing = self.compute_second_deriv_gpu_non_shift(
            transfer_to_host=False)
        deriv2_timing.print_timings()

        d2fdx2_filtered_gpuarray, thresh_timing = self.threshold_above_gpu(
                                                threshold_value=d2fdx2_max,
                                                gpu_array=d2fdx2_gpuarray,
                                                transfer_to_host=True)
        thresh_timing.print_timings()

        # # Find the locations of the filtered derivatives, suspected to be r peaks
        # r_peaks, r_peaks_runtime = self.find_r_peaks(
        #     r_binary_array=d2fdx2_filtered_gpuarray)
        #
        # hr_gpuarray, hr_timing = self.rr_to_hr_gpu(
        #     rr_peaks=r_peaks, transfer_to_host=True)
        #
        # hr_timing.print_timings()
        #
        # hr_thresh_time = self.threshold_hr_values_cpu(heart_rate_data=hr_gpuarray,
        #                                               r_peak_array=r_peaks,
        #                                               min_heart_rate=30.0,
        #                                               max_heart_rate=200.0)
        #
        # print("Total Time: ", deriv_timing.total_time + thresh_timing.total_time +
        #       hr_timing.total_time + hr_thresh_time)

    def plot_lead(self, start=0, stop=1000):
        """
        Plot the lead ECG signal

        :param start: starting index to plot
        :param stop: stop index to plot
        """

        # Try to import matplotlib
        try:
            # import matplotlib
            # matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed")
            return

        print("Plotting lead", self.number, "...")

        to_millivolts = self.resolution / 1000000.0
        plt.plot(self.time.astype(datetime)[start:stop],
                 self.voltage_data[start:stop] * to_millivolts)

        plt.title('ECG Signal: Lead ' + str(self.number))
        plt.xlabel('s')
        plt.ylabel('mV')
        plt.ylim(-1, 1)
        plt.show()

    def save_ecg_voltage_pickle(self, filename=None):
        """
        Save the ecg data as a numpy pickle file
        :param filename: Name of the pickle file (Defaults to lead_N.npy)
        :return: Time to save to file
        """

        if filename is None:
            filename = 'ecg_lead_' + str(self.number)

        print("Attempting to save the lead", self.number, " ecg data to file...")
        start_t = time.clock()
        np.savez(filename,
                 lead=self.number,
                 ecg_data=self.voltage_data)
        end_t = time.clock()
        print('Saved to ', filename)

        return end_t - start_t

    @staticmethod
    def load_ecg_voltage_pickle(filename=None):
        """
        Load the lead ecg data numpy pickle file
        :param filename: name of the file to open
        :return lead number (int), voltage array (float array)
        """

        if filename is None:
            print("Error: load_ecg_voltage_pickle; No file name given..")
            return

        print("Reading ecg voltage from", filename)
        with np.load(filename) as f:
            return f['lead'], f['ecg_data']

    def save_first_derivative_pickle(self, filename=None):
        """
        Save the derivative array to a numpy pickle file
        :param filename: Name of the pickle file
        :return: time used to write to file
        """

        if filename is None:
            filename = 'ecg_lead_1st_deriv' + str(self.number)

        print("Attempting to save lead", self.number, "first derivative to file...")
        start_t = time.clock()
        np.savez(filename,
                 lead=self.number,
                 first_deriv_data=self.voltage_first_derivative)
        end_t = time.clock()
        print('Saved to ', filename)

        return end_t - start_t

    @staticmethod
    def load_first_derivative_pickle(filename=None):
        """
        Load the lead ecg data numpy pickle file
        :param filename: name of the file to open
        """

        if filename is None:
            print("Error: load_first_derivative_pickle; No file name given..")
            return

        print("Reading ecg voltage from", filename)
        with np.load(filename) as f:
            return f['lead'], f['first_deriv_data']

    def save_second_derivative_pickle(self, filename=None):
        """
        Save the derivative array to a numpy pickle file
        :param filename: Name of the pickle file
        :return: time used to write to file
        """

        if filename is None:
            filename = 'ecg_lead_2nd_deriv' + str(self.number)

        print("Attempting to save lead", self.number, "second derivative to file...")
        start_t = time.clock()
        np.savez(filename,
                 lead=self.number,
                 second_deriv_data=self.voltage_second_derivative)
        end_t = time.clock()
        print('Saved to ', filename)

        return end_t - start_t

    @staticmethod
    def load_second_derivative_pickle(filename=None):
        """
        Load the lead 2nd deriv data numpy pickle file
        :param filename: name of the file to open
        """

        if filename is None:
            print("Error: load_second_derivative_pickle; No file name given..")
            return

        print("Reading ecg voltage from", filename)
        with np.load(filename) as f:
            return f['lead'], f['second_deriv_data']

    def save_heart_rate_pickle(self, filename=None):
        """
        Save the derivative array to a numpy pickle file
        :param filename: Name of the pickle file
        :return: time used to write to file
        """

        if filename is None:
            filename = 'heart_rate' + str(self.number)

        print("Attempting to save lead", self.number, "heart rate to file...")
        start_t = time.clock()
        np.savez(filename,
                 lead=self.number,
                 time=self.heart_rate[0],
                 heart_rate=self.heart_rate[1])
        end_t = time.clock()
        print('Saved to ', filename)

        return end_t - start_t

    @staticmethod
    def load_heart_rate_pickle(filename=None):
        """
        Load the lead heart rate data numpy pickle file
        :param filename: name of the file to open
        """

        if filename is None:
            print("Error: load_second_derivative_pickle; No file name given..")
            return

        print("Reading ecg voltage from", filename)
        with np.load(filename) as f:
            return f['lead'], f['time'], f['heart_rate']


class Patient:
    """
    Class that handles the patient info and ecg files
    """

    def __init__(self):
        self.magic_number = None
        self.checksum = None
        self.var_length_block_size = None
        self.sample_size_ecg = None
        self.offset_var_length_block = None
        self.offset_ecg_block = None
        self.file_version = None
        self.first_name = None
        self.last_name = None
        self.ID = None
        self.sex = None
        self.race = None
        self.birth_date = None
        self.record_date = None
        self.file_date = None
        self.start_time = None
        self.n_leads = None
        self.lead_spec = None
        self.lead_quality = None
        self.resolution = None  # Resolution of the amplitude in nanovolts
        self.pacemaker = None
        self.recorder = None
        self.sampling_rate = None  # Hertz
        self.proprietary = None
        self.copyright = None
        self.reserved = None
        self.dt = None
        self.var_block = None
        self.ecg_file = None

        self.samples_per_lead = None

        self.active_leads = None
        self.ecg_lead_derivatives = {}
        self.ecg_lead_voltages = {}
        self.ecg_time_data = None
        self.ecg_data_loaded = False
        self.heart_rate = {}
        self.lead_rr = {}

        # Array of lead objects
        self.leads = []

        self.dt_datetime = None
        self.heart_rate_computed = {}

    def load_ecg_header(self, filename):
        """
        Open the ECG file and only read the header
        :param filename: Name of the ECG file
        :return:
        """


        try:
            with open(filename, 'rb') as self.ecg_file:
                print("Reading filename (header only): " + filename)

                self._get_header_data()

        except IOError:
            print("File cannot be opened:", filename)

    def load_ecg_data(self, filename):
        """
        Open the ECG file and read the data
        :param filename: path name of the file to read
        """

        try:
            with open(filename, 'rb')as self.ecg_file:
                print("Reading filename (header and data): " + filename)

                self._get_header_data()

                # Set the data type to load all of the samples in one chunk
                ecg_dtype = np.dtype([('samples', np.int16, self.n_leads)])

                # Read the file
                ecg_data = np.fromfile(
                    self.ecg_file, dtype=ecg_dtype, count=int(self.samples_per_lead))

                # Put the ecg data into a dictionary
                # for index, lead in np.ndenumerate(self.lead_spec):
                #     if lead == 1:
                #         self.ecg_lead_voltages[index[0]] = ecg_data['samples'][:, index[0]]
                #     else:
                #         self.ecg_lead_voltages[index[0]] = None

                self.active_leads = [i for i, x in enumerate(self.lead_spec) if x == 1]

                t = np.datetime64(str(self.record_date[2]) + '-' +
                                  '{0:02d}'.format(self.record_date[1]) + '-' +
                                  '{0:02d}'.format(self.record_date[0]) + 'T' +
                                  '{0:02d}'.format(self.start_time[0]) + ':' +
                                  '{0:02d}'.format(self.start_time[1]) + ':00.000')

                self.dt_datetime = np.arange(start=t,
                                             stop=t + np.timedelta64(
                                                  int(self.samples_per_lead) * 5, 'ms'),
                                             step=np.timedelta64(5, 'ms'))

                for l in self.active_leads:
                    self.heart_rate_computed[l] = False
                    self.heart_rate[l] = None

                    self.leads.append(
                            Lead(lead_number=l,
                                 voltage=ecg_data['samples'][:, l],
                                 resolution=self.resolution[l],
                                 sample_rate=self.sampling_rate,
                                 samples=self.samples_per_lead,
                                 time_series=self.dt_datetime))

                self.ecg_data_loaded = True

                self.ecg_time_data = np.linspace(0, self.samples_per_lead / self.sampling_rate,
                                                 num=self.samples_per_lead)


        except IOError:
            print("File cannot be opened:", filename)

    def _get_header_data(self):
        self.magic_number = np.fromfile(
            self.ecg_file, dtype=np.dtype('a8'), count=1)[0]
        self.checksum = np.fromfile(self.ecg_file, dtype=np.uint16, count=1)[0]
        self.var_length_block_size = np.fromfile(
            self.ecg_file, dtype=np.int32, count=1)[0]
        self.sample_size_ecg = np.fromfile(
            self.ecg_file, dtype=np.int32, count=1)[0]
        self.offset_var_length_block = np.fromfile(
            self.ecg_file, dtype=np.int32, count=1)[0]
        self.offset_ecg_block = np.fromfile(
            self.ecg_file, dtype=np.int32, count=1)[0]
        self.file_version = np.fromfile(
            self.ecg_file, dtype=np.int16, count=1)[0]
        self.first_name = np.fromfile(
            self.ecg_file, dtype=np.dtype('a40'), count=1)[0]
        self.last_name = np.fromfile(
            self.ecg_file, dtype=np.dtype('a40'), count=1)[0]
        self.ID = np.fromfile(self.ecg_file, dtype=np.dtype('a20'), count=1)[0]
        self.sex = np.fromfile(self.ecg_file, dtype=np.int16, count=1)[0]
        self.race = np.fromfile(self.ecg_file, dtype=np.int16, count=1)[0]
        self.birth_date = np.fromfile(self.ecg_file, dtype=np.int16, count=3)
        self.record_date = np.fromfile(self.ecg_file, dtype=np.int16, count=3)
        self.file_date = np.fromfile(self.ecg_file, dtype=np.int16, count=3)
        self.start_time = np.fromfile(self.ecg_file, dtype=np.int16, count=3)
        self.n_leads = np.fromfile(self.ecg_file, dtype=np.int16, count=1)[0]
        self.lead_spec = np.fromfile(self.ecg_file, dtype=np.int16, count=12)
        self.lead_quality = np.fromfile(self.ecg_file, dtype=np.int16, count=12)
        self.resolution = np.fromfile(self.ecg_file, dtype=np.int16, count=12)
        self.pacemaker = np.fromfile(self.ecg_file, dtype=np.int16, count=1)[0]
        self.recorder = np.fromfile(
            self.ecg_file, dtype=np.dtype('a40'), count=1)[0]
        self.sampling_rate = np.fromfile(
            self.ecg_file, dtype=np.int16, count=1)[0]
        self.proprietary = np.fromfile(
            self.ecg_file, dtype=np.dtype('a80'), count=1)[0]
        self.copyright = np.fromfile(
            self.ecg_file, dtype=np.dtype('a80'), count=1)[0]
        self.reserved = np.fromfile(
            self.ecg_file, dtype=np.dtype('a88'), count=1)[0]
        if self.var_length_block_size > 0:
            self.dt = np.dtype((str, self.var_length_block_size))
            self.var_block = np.fromfile(self.ecg_file, dtype=self.dt, count=1)[0]
        self.samples_per_lead = self.sample_size_ecg / self.n_leads

    def plot_ecg_leads_voltage(self, leads=None, start=0, stop=600):

        """
        Plot the ecg leads (0 ordered!). Defaults to plot all of the leads
        :param leads: nothing (default to 1), integer of lead number, or list of ints
        :param start: starting time to plot in seconds (default 0)
        :param end: ending time to plot (default 5)
        """

        print("Attempting to plot ecg lead voltages...")

        # Try to import matplotlib
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed")
            return

        # Check to make sure ecg data is loaded
        if not self.ecg_data_loaded:
            print("ECG data not loaded yet...")
            return

        if leads is None:
            for lead in self.active_leads:
                l = self.leads[lead]
                to_millivolts = l.resolution / 1000000.0

                print("Plotting lead: " + str(lead))
                plt.plot(l.time.astype(datetime)[start:stop],
                         l.voltage_data[start:stop] * to_millivolts,
                         label="Lead" + str(l.number))

                plt.title("ECG Lead Voltages")
                plt.xlabel("s")
                plt.ylabel("mV")
                plt.ylim(-1, 1)
                plt.legend()
            plt.show()

        else:
            for lead in leads:
                if lead not in self.active_leads:
                    print("Lead", lead, "not a valid lead number")
                    print("Valid leads", self.active_leads)
                    continue

                l = self.leads[lead]
                to_millivolts = l.resolution / 1000000.0

                print("Plotting lead: " + str(lead))
                plt.plot(l.time.astype(datetime)[start:stop],
                         l.voltage_data[start:stop] * to_millivolts,
                         label="Lead" + str(l.number))

                plt.title("ECG Lead Voltages")
                plt.xlabel("s")
                plt.ylabel("mV")
                plt.ylim(-1, 1)
                plt.legend()
            plt.show()

        print("Done...")

    def compute_hr_gpu(self, lead_number=None):
        """ 
        Compute the derivatives of a given lead signal
        :param lead_number: Integer of the lead number to calculate the derivative of
        :return: Returns an array of the derivatives from the specified lead number
        """

        # Try to import the pyCUDA module
        try:
            from pycuda import driver, compiler, gpuarray, tools, cumath
            import pycuda.autoinit
            from pycuda.compiler import SourceModule
        except ImportError:
            print("Error: compute_hr_gpu(); Unable to import the pyCUDA package")
            return

        # Check to make sure the given lead number is valid
        if self.lead_spec[lead_number] == 1:
            pass
        else:
            print("Error: compute_hr_gpu(); Given lead number", lead_number, "is not valid ")
            print("Valid lead numbers: ", [i for i, x in enumerate(self.lead_spec) if x == 1])
            return

        print()
        print("---------------------------------------------")
        print("Computing heart rate for lead", lead_number, "on the gpu...")
        print("---------------------------------------------")
        print()

        lead_voltage = self.ecg_lead_voltages[lead_number] * (self.resolution[lead_number] / 1000000.0)
        lead_voltage = lead_voltage.astype(np.float32)
        dx = self.samples_per_lead / self.sampling_rate / self.samples_per_lead

        # Create the gpu arrays
        # 1st deriv
        # lead_voltage_gpu = gpuarray.to_gpu(lead_voltage[:-1])
        # lead_voltage_gpu_shifted = gpuarray.to_gpu(lead_voltage[1:])

        total_t = 0
        start_t_gpu = driver.Event()
        end_t_gpu = driver.Event()

        # 2nd deriv
        start_t_gpu.record()

        lead_voltage_gpu = gpuarray.to_gpu(lead_voltage[:-2])
        lead_voltage_gpu_shifted = gpuarray.to_gpu(lead_voltage[1:-1])
        lead_voltage_gpu_shifted_again = gpuarray.to_gpu(lead_voltage[2:])

        end_t_gpu.record()
        end_t_gpu.synchronize()
        lead_t = start_t_gpu.time_till(end_t_gpu)*1e-3
        total_t += lead_t
        print("Transfer Lead to GPU: \t\t\t{0:.5f} seconds".format(lead_t))

        alpha = np.float32(1.0 / dx ** 2)

        n = lead_voltage[2:].size

        dfdx_gpu = gpuarray.zeros(n, np.float32)
        rr_binary_array_gpu = gpuarray.zeros(n, np.int32)

        # Device info
        gpu_dev = tools.DeviceData()
        threads_per_block = 512  # gpu_dev.max_threads
        n_blocks = np.int_(np.ceil(n / threads_per_block))

        # CUDA kernels
        #with open('kernels.cu', 'r') as f:
        module = SourceModule(
            """
            #include <stdio.h>
            // Simple gpu kernel to compute the derivative with a constant dx (using shared memory)
            __global__ void dfdx_shared(float* f, float* f_shifted, float* dfdx, float dx, int n)
            {
                unsigned int g_i = blockIdx.x * blockDim.x + threadIdx.x;
                unsigned int s_i = threadIdx.x;

                __shared__ float shared_dx;
                __shared__ float shared_f[512];
                __shared__ float shared_f_shifted[512];

                // Assigned to shared memory
                shared_dx = dx;
                shared_f[s_i] = f[g_i];
                shared_f_shifted[s_i] = f_shifted[g_i];

                __syncthreads();

                dfdx[g_i] = (shared_f_shifted[s_i] - shared_f[s_i]) / shared_dx;
            }

            // Simple gpu kernel to compute the derivative with a constant dx
            __global__ void dfdx(float* f, float* f_shifted, float* dfdx, float dx, int n)
            {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                dfdx[i] = (f_shifted[i] - f[i]) / dx;
            }

            // Simple gpu kernel to compute the derivative with a constant dx
            __global__ void d2fdx2(float* f, float* f_s, float* f_ss, float* d2fdx2, float alpha, int n)
            {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                d2fdx2[i] = alpha * (f[i] - 2 * f_s[i] + f_ss[i]);
            }

            // Simple gpu kernel to compute the derivative with a constant dx
            __global__ void d2fdx2_shared(float* f, float* f_s, float* f_ss, float* d2fdx2, float alpha, int n)
            {
                unsigned int g_i = blockIdx.x * blockDim.x + threadIdx.x;
                unsigned int s_i = threadIdx.x;

                __shared__ float shared_alpha;
                __shared__ float shared_f[512];
                __shared__ float shared_f_s[512];
                __shared__ float shared_f_ss[512];

                // Assigned to shared memory
                shared_alpha = alpha;
                shared_f[s_i] = f[g_i];
                shared_f_s[s_i] = f_s[g_i];
                shared_f_ss[s_i] = f_ss[g_i];

                __syncthreads();

                d2fdx2[g_i] = shared_alpha * (shared_f[s_i] - 2 * shared_f_s[s_i] + shared_f_ss[s_i]);
            }

            // Filter out values that are not big enough (use for very large derivatives) i.e. large positive spikes
            __global__ void get_only_large_deriv_shared(float* f, int* thresholded_f, float threshold)
            {
                __shared__ float shared_f[512];
                __shared__ int  shared_thresholded_f[512];
                //float shared_threshold;

                unsigned int g_i = blockIdx.x * blockDim.x + threadIdx.x;
                unsigned int b_i = threadIdx.x;

                //shared_threshold = threshold;

                // Put f into shared memory
                shared_f[b_i] = f[g_i];

                __syncthreads();

                // 1 if at or above threshold, 0 if not
                shared_thresholded_f[b_i] = (shared_f[b_i] >= threshold) ? 1 : 0;

                // Put threshold values into global memory
                thresholded_f[g_i] = shared_thresholded_f[b_i];

            }

            // Filter out values that are not big enough (use for very large derivatives) i.e. large positive spikes
            __global__ void get_only_large_deriv(float* f, int* thresholded_f, float threshold)
            {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

                // 1 if at or above threshold, 0 if not
                thresholded_f[i] = (f[i] >= threshold) ? 1 : 0;
                //printf("%f vs %f - > %i \\n", f[i], threshold, thresholded_f[i]);
            }

            __global__ void filter_bad_hr(float* hr, float* hr_filtered, float threshold)
            {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

                // Turn bad heart rate values go to 0
                hr_filtered[i] = (hr[i] >= threshold) ? hr[i] : 0;
            }

            __global__ void rr_peaks(float* r_peaks,
                                     float* r_peaks_shifted,
                                     float* hr,
                                     float alpha)
            {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

                // alpha is 60/dt
                // hr is the distance between peaks div * 2 / by time
                // i.e. 2 beats (distance/time) = beats per time

                hr[i] = alpha / (r_peaks_shifted[i] - r_peaks[i]);
                //hr[i] = 60.0 / (r_peaks_shifted[i] - r_peaks[i]);
                //printf("alpha %f, %f, %f, heart rate %f\\n", alpha, r_peaks[i], r_peaks_shifted[i], hr[i]);
                //printf("%i, %f, %f, heart rate %f\\n",i, r_peaks[i], r_peaks_shifted[i], hr[i]);
            }
            """)
        # print('Launching', n_blocks, 'blocks with', threads_per_block, 'threads per block for a total of',
        #      n_blocks * threads_per_block, 'threads')

        # Calculate the first derivative
        derivative_kernel = module.get_function("d2fdx2_shared")

        # print("Calling derivative kernel...")
        # derivative_kernel(lead_voltage_gpu,
        #                   lead_voltage_gpu_shifted,
        #                   dfdx_gpu,
        #                   dx,
        #                   np.int_(n),
        #                   block=(threads_per_block, 1, 1),
        #                   grid=(n_blocks, 1, 1))

        # print("Calling derivative kernel...")



        start_t_gpu.record()
        derivative_kernel(lead_voltage_gpu,
                          lead_voltage_gpu_shifted,
                          lead_voltage_gpu_shifted_again,
                          dfdx_gpu,
                          alpha,
                          np.int_(n),
                          block=(threads_per_block, 1, 1),
                          grid=(n_blocks, 1, 1))
        end_t_gpu.record()
        end_t_gpu.synchronize()
        deriv_t = start_t_gpu.time_till(end_t_gpu)*1e-3
        total_t += deriv_t
        print("Derivative GPU Kernel: \t\t\t{0:.5f} seconds".format(deriv_t))
        # print("Done")
        # thresh = 0.7

        # max_dfdx = gpuarray.max(dfdx_gpu).get()
        # min_dfdx = gpuarray.min(dfdx_gpu)

        # print("Max dfdx", max_dfdx)

        # threshold_value = np.float32(thresh) * max_dfdx
        threshold_value = np.float32(18000.0)
        # print("Using", threshold_value, "to threshold")

        # Get the large positive derivative spikes
        threshold_kernel = module.get_function("get_only_large_deriv_shared")

        # print("Calling derivative filter kernel...")
        start_t_gpu.record()
        threshold_kernel(dfdx_gpu,
                         rr_binary_array_gpu,
                         threshold_value,
                         block=(threads_per_block, 1, 1),
                         grid=(n_blocks, 1, 1))

        end_t_gpu.record()
        end_t_gpu.synchronize()
        thresh_t = start_t_gpu.time_till(end_t_gpu)*1e-3
        total_t += thresh_t
        print("Derivative Filter GPU Kernel: \t\t{0:.5f} seconds".format(thresh_t))

        # Get the r_peaks locations
        start_t_gpu.record()

        rr_binary_array_cpu = rr_binary_array_gpu.get()

        end_t_gpu.record()
        end_t_gpu.synchronize()
        thresh_t_transfer = start_t_gpu.time_till(end_t_gpu)*1e-3
        total_t += thresh_t_transfer
        print("Transfer Filtered Deriv. GPU to CPU: \t{0:.5f} seconds".format(thresh_t_transfer))

        # Find the locations of the filtered derivatives, suspected to be r peaks
        start_t_cpu = time.clock()

        rr_peaks = np.ascontiguousarray(np.where(rr_binary_array_cpu != 0)[0])

        end_t_cpu = time.clock()
        total_t += end_t_cpu - start_t_cpu
        print("Locate R peaks on CPU: \t\t\t{0:.5f} seconds".format(end_t_cpu - start_t_cpu))

        # GPU Arrays of the r peaks
        start_t_gpu.record()

        rr_peaks_gpu = gpuarray.to_gpu(rr_peaks[:-1].astype(np.float32))
        rr_peaks_gpu_shifted = gpuarray.to_gpu(rr_peaks[1:].astype(np.float32))

        end_t_gpu.record()
        end_t_gpu.synchronize()
        thresh_t_transfer = start_t_gpu.time_till(end_t_gpu)*1e-3
        total_t += thresh_t_transfer
        print("Transfer R peaks CPU to GPU: \t\t{0:.5f} seconds".format(thresh_t_transfer))


        # GPU array to hold the heart rate
        # print("rr_peaks.size", rr_peaks.size)
        if rr_peaks.size == 0:
            print("Error, finding peaks failed for lead", lead_number, "...")
            return

        hr_gpu = gpuarray.zeros(rr_peaks.size, np.float32)

        # Get the distances between each peak and find the rate
        rr_peaks_kernel = module.get_function("rr_peaks")

        # 2 beats per time interval to get a heart rate
        # Send a single float instead of calculating every time on the gpu
        alpha = np.float32(60.0 / dx)

        n_blocks = np.int_(np.ceil(rr_peaks.size / threads_per_block))

        # print("Calling rr peaks kernel...")
        start_t_gpu.record()

        rr_peaks_kernel(rr_peaks_gpu,
                        rr_peaks_gpu_shifted,
                        hr_gpu,
                        alpha,
                        block=(threads_per_block, 1, 1),
                        grid=(n_blocks, 1, 1))

        end_t_gpu.record()
        end_t_gpu.synchronize()
        rr_t = start_t_gpu.time_till(end_t_gpu)*1e-3
        total_t += rr_t
        print("Heart Rate Calc GPU Kernel: \t\t{0:.5f} seconds".format(rr_t))
        # print("Done...")

        # Get the max heart rate value and get a threshold of the max
        # max_hr = gpuarray.max(hr_gpu).get()
        # print('max_hr', max_hr)
        hr_min = np.float32(30.0)
        hr_max = np.float32(200.0)
        # print("Filtering out bad heart rates between", hr_min,"and",hr_max, "...")

        # Get the heart rate data from the gpu
        start_t_gpu.record()

        hr_cpu = hr_gpu.get()

        end_t_gpu.record()
        end_t_gpu.synchronize()
        hr_transfer_to_h = start_t_gpu.time_till(end_t_gpu)*1e-3
        total_t += hr_transfer_to_h
        print("Transfer HR GPU to CPU: \t\t{0:.5f} seconds".format(hr_transfer_to_h))

        # Find where the heart rate is high enough to be valid
        start_t_cpu = time.clock()
        good_indices = np.where((hr_cpu >= hr_min) & (hr_cpu <= hr_max))[0]

        filtered_hr = np.zeros(good_indices.size, dtype='float32')
        filtered_t = np.zeros(good_indices.size, dtype='float32')

        for i, v in np.ndenumerate(good_indices):

            # Get the filtered heart rate value
            filtered_hr[i] = hr_cpu[good_indices[i]]

            # Get the corresponding time that the heart rate occurs
            filtered_t[i] = rr_peaks[good_indices[i]]

        end_t_cpu = time.clock()

        filter_t = end_t_cpu - start_t_cpu
        print("Heart Rate Filtering on CPU: \t\t{0:.5f} seconds".format(filter_t))
        total_t += filter_t
        # Save it to the object
        self.heart_rate[lead_number] = [None, None]
        self.heart_rate[lead_number][1] = np.copy(filtered_hr)
        self.heart_rate[lead_number][0] = np.copy(filtered_t)

        #print("Done...")
        # Save it to a file
        np.savez('hr' + str(lead_number) + '.npz', hr=filtered_hr, t=filtered_t)

        print("-------------------------")
        print("Total time: \t\t\t\t{0:.5f} seconds".format(total_t))
        print()
        print()

        self.heart_rate_computed[lead_number] = True

    def find_rr_peaks_josh(self, lead_number=None):
        """
        """

        print("Finding rr peaks for lead number", lead_number, "...")

        df = self.ecg_lead_derivatives[lead_number]

        dfm1 = 0
        large = 0
        loc = 0
        # if(abs(max(df))>abs(min(df))):
        #	thresh = max(df)*.7
        # else:
        #	thresh = min(df)*.7
        peaks = []
        peaks_location = []
        last = 0
        twoago = 0
        k = 0
        for i in df:
            if (k % 100000 == 0) & (len(df) - (100000 + k) >= 0):
                for x in range(0, 100000):
                    if abs(df[k + x]) > abs(large):
                        large = df[k + x]
                        thresh = df[k + x] * .7
                        # print("Peaks with threshold " + repr(thresh) + ":\n")
            if abs(last) >= abs(thresh):
                if abs(last) > abs(twoago):
                    if abs(last) > abs(i):
                        peaks.append(last)
                        peaks_location.append(loc)
                        # print(repr(last) + " : " + repr(loc) + "\n")
            twoago = last
            last = i
            k += 1
            large = 0;
            loc += (self.samples_per_lead / self.sampling_rate) / self.samples_per_lead
        RR = 0
        HR = []
        last = 0
        k = 0

        for i in peaks_location:
            if k > 0:
                if ((60 / (i - last)) > 200) | ((60 / (i - last)) < 30):
                    HR.append(np.nan)
                else:
                    HR.append(60 / (i - last))
            # print(repr (i) + "-" + repr(last) + "=" + repr((i-last)) + "\n")
            k += 1
            last = i
        xh = np.linspace(0, 3600 * 24, num=len(HR))

        self.heart_rate[lead_number] = [None, None]
        self.heart_rate[lead_number][1] = np.copy(np.array(HR))
        self.heart_rate[lead_number][0] = np.copy(xh)

    def plot_hr_data(self):
        """
        Plot the heart rate data
        :return:
        """

        import scipy.signal

        # Try to import matplotlib
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed")
            return

        plot_me = False
        for lead_number in self.active_leads:
            if self.heart_rate_computed[lead_number] is True:
                hr = self.heart_rate[lead_number][1]
                hr_t = self.heart_rate[lead_number][0]

                print("Performing median filter on lead", lead_number, "...")
                filtered_hr = scipy.signal.medfilt(hr, (125,))
                print("Done")

                plt.plot(hr_t, filtered_hr, label='lead ' + str(lead_number))
                plt.title('Heart Rate (BPM) vs Time')
                plt.xlabel('Time')
                plt.ylabel('BPM')
                plt.legend()
                plot_me = True
            else:
                print("Heart rate data not computed for lead", lead_number)

        if plot_me:
            plt.show()

    @staticmethod
    def get_window_best_rr_distance(ary):
        """
        Find the best rr peak
        """
        distances = np.zeros(ary.size)
        for i, value in np.ndenumerate(ary):
            a_i = i[0]
            if a_i > 0:
                distances[a_i-1] = ary[a_i] - ary[a_i-1]

        hist, bins = np.histogram(distances, bins=20, range=(0,200))
        #print('bins',bins)
        #print('hist', hist)
        #print('max hist', np.max(hist), 'at index', np.argmax(hist[1:]), 'value',bins[np.argmax(hist[1:])])

        #width = 0.7 * (bins[1] - bins[0])
        #center = (bins[:-1] + bins[1:]) / 2
        #plt.bar(center, hist, align='center', width=width)
        #plt.show()
        #mu, sigma = np.mean(distances), np.std(distances)

        return bins[np.argmax(hist[1:])]

    def filter_hr_history(self, lead_number=None):
        """
        Filter the input array
        :return: Filtered array
        """
        import scipy.signal

        return scipy.signal.medfilt(self.heart_rate[lead_number][1], (195,))

    def plot_ecg_data_bokeh(self):
        from bokeh.io import show
        from bokeh.plotting import figure
        from bokeh.palettes import Spectral11
        palette = Spectral11

        bp = figure()
        bp.xaxis.axis_label = "t"
        bp.yaxis.axis_label = "mV"
        s_interval = 0
        e_interval = 500

        for i in self.active_leads:
            bp.line(y=self.ecg_lead_derivatives[i][s_interval:e_interval],
                    x=self.ecg_time_data[s_interval+1:e_interval + 1],
                    line_color=palette[i])
        show(bp)

