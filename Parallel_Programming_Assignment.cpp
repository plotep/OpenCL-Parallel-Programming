

// Parallel Programming Assignment 1
// Name : Pawel Bielinski
// Student ID : BIE18679372
// Email : BIE18679372@students.lincoln.ac.uk
// The code implements the basic functionalities of calculating mean, min maxand standard deviation for int variables, as well as
// for floats. This is done using reduction, however a scan implementation is also included. 
// The hillis-steele implementation for mean calculation and the structure of the program has been adapted from Tutorial 3 ref - https://github.com/alanmillard/OpenCL-Tutorials
// Atomic functions have been made for the float values, adapted from : https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/ .
// Sorting was implement using the selection sort algorithm, found on : http://www.bealto.com/gpu-sorting_parallel-selection.html . This has been changed so that it 
// fits the structure of the temperature statistics program. From the sorted output array, the quartile values are calculated and shown. Kernel runtimes are reported
// for both variable types, the memory buffer transfer time is recorded and shown in the console output, and the total int and float operation runtimes are recorded.
//  The total program runtime is also recorded and shown.
// 


#include <iostream>
#include <vector>
#include <algorithm>
#include "Utils.h"
void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	// ============================INIT====================================================INIT====================================================INIT========================
	// Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0;}
	}

	// detect any potential exceptions
	try {
		// Part 2 - host operations
		// 2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		// display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		// create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// 2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		// build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		// ============================INIT====================================================INIT====================================================INIT========================

		typedef float typefloat;
		typedef int typeint;

		// vector for the temp file 
		std::vector<string> temperature_file_raw;	
		// vector for just degree values from file, float
		std::vector<typefloat> temperature_vals_float;	
		// vector for just degree values from file , int
		std::vector<typeint> temperature_vals_int;
		// file location
		string filename = "../temp_lincolnshire_short.txt";
		std::cout << "FILE = :  " << filename << endl << endl;
		std::cout << "Calculating Statistics...	" << endl;
		// temporary value store
		string temp_val;
		// filestream
		ifstream myReadFile;
		// opening file
		myReadFile.open(filename);
		// reading in values from txt file
		if (myReadFile.is_open()) {
			while (!myReadFile.eof()) {
				myReadFile >> temp_val;
				// reading in values from text file to vector
				temperature_file_raw.push_back(temp_val);
			}
		}
		// closing file once finished
		myReadFile.close();
		// extracting temp vals
		for (int i = 5; i < temperature_file_raw.size(); i += 6)
		{
			// converting temperature strings into floats
			float temp = strtof(temperature_file_raw[i].c_str(), 0);
			// copying the temp floats to a 1d vector array
			temperature_vals_float.push_back(temp);
			temperature_vals_int.push_back(temp);
		}
		// number of workers
		size_t workers = 64;
		// size in bytes
		size_t workers_bytes = workers * sizeof(typefloat);
		// size in bytes
		size_t workers_int_bytes = workers * sizeof(typeint);
		// size of temperature array used to calculate mean values
		int size_of_temperature_array = temperature_vals_float.size();
		//padding size
		size_t padding_size = temperature_vals_float.size() % workers;
		// if the input vector is not a multiple of the workers
		// insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			// create an extra vector with neutral values
			std::vector<typefloat> temperature_append(workers-padding_size, 0);
			std::vector<typeint> temperature_append_int(workers - padding_size, 0);
			// append that extra vector to our input
			temperature_vals_float.insert(temperature_vals_float.end(), temperature_append.begin(), temperature_append.end());
			temperature_vals_int.insert(temperature_vals_int.end(), temperature_vals_int.begin(), temperature_vals_int.end());
		}
		// size of padded array
		size_t size_of_temperature_array_padded = temperature_vals_float.size();
		// input size in bytes float
		size_t input_size = temperature_vals_float.size()*sizeof(typefloat);
		// input size in bytes int
		size_t input_size_int = temperature_vals_int.size() * sizeof(typeint);
		// number of worker groups 
		size_t nr_groups = size_of_temperature_array_padded / workers;

		//---------------------------------------------------ARRAYTOBUFFERINT---------------------------------------------------------------

		// array copy time profiler 
		cl::Event copy_buffer_array_int_profile;
		// input buffer 
		cl::Buffer input_buffer_int(context, CL_MEM_READ_ONLY, input_size_int);
		// copy temp array to buffer
		queue.enqueueWriteBuffer(input_buffer_int, CL_TRUE, 0, input_size_int, &temperature_vals_int[0], NULL, &copy_buffer_array_int_profile);

		//---------------------------------------------------ARRAYTOBUFFERFLOAT-------------------------------------------------------------
													 
		// input profiler 
		cl::Event copy_buffer_array_profile;
		// input buffer 
		cl::Buffer input_buffer_float(context, CL_MEM_READ_ONLY, input_size);
		// copy temp array to buffer
		queue.enqueueWriteBuffer(input_buffer_float, CL_TRUE, 0, input_size, &temperature_vals_float[0], NULL, &copy_buffer_array_profile);

		// ---------------------------------------------------AVERAGEINT-------------------------------------------------------------

		// output array, size 1 as the items are added together atomically in kernel
		std::vector<typeint> sum_output_int(1);
		// size of sum output in bytes
		size_t output_size_int = sum_output_int.size() * sizeof(typeint);
		// kernel profiler
		cl::Event sum_kernel_profiling_event_int;
		// buffer for reduced sums
		cl::Buffer sum_output_buffer_int(context, CL_MEM_READ_WRITE, output_size_int);
		// initializing buffer
		queue.enqueueFillBuffer(sum_output_buffer_int, 0, 0, output_size_int);
		// initializing kernel 
		cl::Kernel kernel_sum_int = cl::Kernel(program, "reduce_add_int");
		// passing input and output buffers
		kernel_sum_int.setArg(0, input_buffer_int);
		kernel_sum_int.setArg(1, sum_output_buffer_int);
		// workers * size of int
		kernel_sum_int.setArg(2, cl::Local(workers_int_bytes));
		// que kernel
		queue.enqueueNDRangeKernel(kernel_sum_int, cl::NullRange, cl::NDRange(size_of_temperature_array_padded), cl::NDRange(workers), NULL, &sum_kernel_profiling_event_int);
		// get kernel output 
		queue.enqueueReadBuffer(sum_output_buffer_int, CL_TRUE, 0, output_size_int, &sum_output_int[0]);
		// get sum from the first array element , divide by total numbers to get mean
		float sum_int = float(sum_output_int[0]) / float(size_of_temperature_array);
		std::cout << "----MEAN_INT---- :	" << sum_int << endl;

		// ---------------------------------------------------AVERAGEFLOAT-------------------------------------------------------------

		// vector for the reduced sums, size determined by number of workgroups
		std::vector<typefloat> sum_output(nr_groups);
		// size of output in bytes
		size_t output_size = sum_output.size() * sizeof(typefloat);
		// kernel profiler
		cl::Event sum_kernel_profiling_event;
		// buffer for reduced sums	
		cl::Buffer sum_output_buffer(context, CL_MEM_READ_WRITE, output_size);
		// initializing buffer
		queue.enqueueFillBuffer(sum_output_buffer, 0, 0, output_size);
		// initializing kernel 
		cl::Kernel kernel_sum = cl::Kernel(program, "reduce_add_float");
		// passing input and output buffers
		kernel_sum.setArg(0, input_buffer_float);
		kernel_sum.setArg(1, sum_output_buffer);
		// workers * size of type
		kernel_sum.setArg(2, cl::Local(workers_bytes));
		// que kernel
		queue.enqueueNDRangeKernel(kernel_sum, cl::NullRange, cl::NDRange(size_of_temperature_array_padded), cl::NDRange(workers),NULL, &sum_kernel_profiling_event);
		// get kernel output 
		queue.enqueueReadBuffer(sum_output_buffer, CL_TRUE, 0, output_size, &sum_output[0]);
		// get reduced sums from workers, add together and divide by total numbers to get mean
		float reducedsums_added = 0;
		// iterate over output arrays, add reduced sums
		for (std::vector<float>::iterator it = sum_output.begin(); it != sum_output.end(); ++it)
			reducedsums_added += *it;
		float sum_float = float(reducedsums_added) / float(size_of_temperature_array);
		std::cout << "----MEAN_FLOAT---- :	" << sum_float << endl;
		//float manualsums = 0;
		//for (std::vector<float>::iterator it = temperature_vals_float.begin(); it != temperature_vals_float.end(); ++it)
		//	manualsums += *it;
		//std::cout << "----MEAN_FLOAT_NON_OPENCL---- :  " << float(manualsums)/ float(size_of_temperature_array) << endl << endl;
		// ---------------------------------------------------AVERAGE_FLOAT_ATOMIC-------------------------------------------------------------


		// vector for the reduced sums, size determined by number of workgroups
		std::vector<typefloat> sum_output_atomic(1);
		// size of output in bytes
		size_t output_size_atomic = sum_output_atomic.size() * sizeof(typefloat);
		// kernel profiler
		cl::Event sum_kernel_profiling_event_atomic;
		// buffer for reduced sums	
		cl::Buffer sum_output_buffer_atomic(context, CL_MEM_READ_WRITE, output_size_atomic);
		// initializing buffer
		queue.enqueueFillBuffer(sum_output_buffer_atomic, 0, 0, output_size_atomic);
		// initializing kernel 
		cl::Kernel kernel_sum_atomic = cl::Kernel(program, "reduce_add_float_atomic");
		// passing input and output buffers
		kernel_sum_atomic.setArg(0, input_buffer_float);
		kernel_sum_atomic.setArg(1, sum_output_buffer_atomic);
		// workers * size of type
		kernel_sum_atomic.setArg(2, cl::Local(workers_bytes));
		// que kernel
		queue.enqueueNDRangeKernel(kernel_sum_atomic, cl::NullRange, cl::NDRange(size_of_temperature_array_padded), cl::NDRange(workers), NULL, &sum_kernel_profiling_event_atomic);
		// get kernel output 
		queue.enqueueReadBuffer(sum_output_buffer_atomic, CL_TRUE, 0, output_size_atomic, &sum_output_atomic[0]);
		// sum of items / number of items
		float sum_float_atomic = float(sum_output_atomic[0]) / float(size_of_temperature_array);
		std::cout << "----MEAN_FLOAT_ATOMIC---- :	" << sum_float_atomic << endl;
		// ---------------------------------------------------AVERAGE_FLOAT_BLELLOCH-------------------------------------------------------------

		// vector for the sums
		std::vector<typefloat> sum_output_blelloch(size_of_temperature_array_padded);
		std::vector<typefloat> sum_output_blelloch2(nr_groups);
		// size of output in bytes
		size_t output_size_blelloch = sum_output_blelloch.size() * sizeof(typefloat);
		size_t output_size_blelloch2 = nr_groups * sizeof(typefloat);
		// kernel profiler
		cl::Event sum_kernel_profiling_event_blelloch;
		// buffer for reduced sums	
		cl::Buffer sum_output_buffer_blelloch(context, CL_MEM_READ_WRITE, output_size_blelloch);
		// initializing buffer
		cl::Buffer sum_output_buffer_blelloch2(context, CL_MEM_READ_WRITE, output_size_blelloch2);
		queue.enqueueFillBuffer(sum_output_buffer_blelloch, 0, 0, output_size_blelloch);
		queue.enqueueFillBuffer(sum_output_buffer_blelloch2, 0, 0, output_size_blelloch2);
		// initializing kernel 
		cl::Kernel kernel_sum_hillis = cl::Kernel(program, "blelloch_scan");
		// passing input and output buffers
		kernel_sum_hillis.setArg(0, input_buffer_float);
		kernel_sum_hillis.setArg(1, sum_output_buffer_blelloch);
		kernel_sum_hillis.setArg(2, sum_output_buffer_blelloch2);
		//// workers * size of type
		kernel_sum_hillis.setArg(3, cl::Local(workers_bytes));
		// que kernel
		queue.enqueueNDRangeKernel(kernel_sum_hillis, cl::NullRange, cl::NDRange(size_of_temperature_array_padded), cl::NDRange(workers), NULL, &sum_kernel_profiling_event_blelloch);
		// get kernel output 
		queue.enqueueReadBuffer(sum_output_buffer_blelloch2, CL_TRUE, 0, output_size_blelloch, &sum_output_blelloch[0]);
		float sumhillis = 0;
		////get partial sums from each workgroup and add them together
		float sum_from_workgroup_float3 = 0;
		for (std::vector<float>::iterator it = sum_output_blelloch.begin(); it != sum_output_blelloch.end(); ++it)
			sum_from_workgroup_float3 += *it;
		float avgee = sum_from_workgroup_float3 / float(size_of_temperature_array);
		std::cout << "----MEAN_FLOAT_BLELLOCH---- :	" << avgee << endl << endl;;

		// ---------------------------------------------------MINIMUM_FLOAT-------------------------------------------------------------

		// output array, size determined by amount of workgroups
		std::vector<typefloat> output_minimum(nr_groups);
		// size of output in bytes
		size_t output_size_min_float = nr_groups * sizeof(typefloat);
		// kernel profiler
		cl::Event min_kernel_profiler;
		// output buffer for min values from each group
		cl::Buffer min_output_buffer(context, CL_MEM_READ_WRITE, output_size_min_float);
		// que kernel
		cl::Kernel kernel_min = cl::Kernel(program, "reduce_min");
		// passing input and output buffers
		kernel_min.setArg(0, input_buffer_float);
		kernel_min.setArg(1, min_output_buffer);
		// workers * size of type
		kernel_min.setArg(2, cl::Local(workers_bytes));
		// que kernel
		queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(size_of_temperature_array_padded), cl::NDRange(workers), NULL, &min_kernel_profiler);
		// get kernel output 
		queue.enqueueReadBuffer(min_output_buffer, CL_TRUE, 0, output_size_min_float, &output_minimum[0]);
		// iterate over the minimum values found through reduction, find smallest
		auto min_value = *std::min_element(output_minimum.begin(), output_minimum.end());
		std::cout << "----MIN_FLOAT---- :	" << min_value << endl;

		// ---------------------------------------------------MINIMUM_FLOAT_ATOMIC-------------------------------------------------------------

		// output array, size determined by amount of workgroups
		std::vector<typefloat> output_minimum_atomic(1);
		// size of output in bytes
		size_t output_size_min_float_atomic = output_minimum_atomic.size() * sizeof(typefloat);
		// kernel profiler
		cl::Event min_kernel_profiler_atomic;
		// output buffer for min values from each group
		cl::Buffer min_output_buffer_atomic(context, CL_MEM_READ_WRITE, output_size_min_float_atomic);
		// que kernel
		cl::Kernel kernel_min_atomic = cl::Kernel(program, "reduce_min_atomic");
		// passing input and output buffers
		kernel_min_atomic.setArg(0, input_buffer_float);
		kernel_min_atomic.setArg(1, min_output_buffer_atomic);
		// workers * size of type
		kernel_min_atomic.setArg(2, cl::Local(workers_bytes));
		// que kernel
		queue.enqueueNDRangeKernel(kernel_min_atomic, cl::NullRange, cl::NDRange(size_of_temperature_array_padded), cl::NDRange(workers), NULL, &min_kernel_profiler_atomic);
		// get kernel output 
		queue.enqueueReadBuffer(min_output_buffer_atomic, CL_TRUE, 0, output_size_min_float_atomic, &output_minimum_atomic[0]);
		// iterate over the minimum values found through reduction, find smallest
		auto min_value_atomic = *std::min_element(output_minimum_atomic.begin(), output_minimum_atomic.end());
		std::cout << "----MIN_FLOAT_ATOMIC---- :	" << min_value_atomic << endl;

		// ---------------------------------------------------MINIMUM_INT-------------------------------------------------------------

		// output array, size 1 as mins are looped over atomically and the min is output in kernel
		std::vector<typeint> output_min_int(1);
		// size of output in bytes
		size_t output_size_min_int = output_min_int.size() * sizeof(typeint);
		// kernel profiler
		cl::Event min_int_kernel_profiler;
		// output buffer for min value
		cl::Buffer min_int_output_buffer(context, CL_MEM_READ_WRITE, output_size_min_int);
		// que kernel
		cl::Kernel kernel_min_int = cl::Kernel(program, "reduce_min_int");
		// passing input and output buffers
		kernel_min_int.setArg(0, input_buffer_int);
		kernel_min_int.setArg(1, min_int_output_buffer);
		// workers * size of type
		kernel_min_int.setArg(2, cl::Local(workers_int_bytes));
		// que kernel
		queue.enqueueNDRangeKernel(kernel_min_int, cl::NullRange, cl::NDRange(size_of_temperature_array_padded), cl::NDRange(workers), NULL, &min_int_kernel_profiler);
		// get kernel output 
		queue.enqueueReadBuffer(min_int_output_buffer, CL_TRUE, 0, output_size_min_int, &output_min_int[0]);
		// getting the minimum value at [0]
		float min_value_int = output_min_int[0];
		std::cout << "----MIN_INT---- :	" << min_value_int << endl << endl;

		// ---------------------------------------------------MAX_FLOAT_ATOMIC-------------------------------------------------------------

		// output array, size determined by amount of workgroups
		std::vector<typefloat> output_maximum_atomic(nr_groups);
		// size of output in bytes
		size_t output_size_max_float_atomic = output_maximum_atomic.size() * sizeof(typeint);
		// kernel profiler
		cl::Event max_kernel_profiler_atomic;
		// reduced max output buffer
		cl::Buffer max_output_buffer_atomic(context, CL_MEM_READ_WRITE, output_size_max_float_atomic);
		// que kernel
		cl::Kernel kernel_max_atomic = cl::Kernel(program, "reduce_max_atomic");
		// passing input and output buffers
		kernel_max_atomic.setArg(0, input_buffer_float);
		kernel_max_atomic.setArg(1, max_output_buffer_atomic);
		// workers * size of type
		kernel_max_atomic.setArg(2, cl::Local(workers_bytes));
		// que kernel
		queue.enqueueNDRangeKernel(kernel_max_atomic, cl::NullRange, cl::NDRange(size_of_temperature_array_padded), cl::NDRange(workers), NULL, &max_kernel_profiler_atomic);
		// get kernel output 
		queue.enqueueReadBuffer(max_output_buffer_atomic, CL_TRUE, 0, output_size_max_float_atomic, &output_maximum_atomic[0]);
		// iterating over the reduced maxes to find the max value
		float max_value_atomic = output_maximum_atomic[0];
		std::cout << "----MAX_FLOAT_atomic---- :	" << max_value_atomic << endl;

		// ---------------------------------------------------MAX_FLOAT-------------------------------------------------------------

		// output array, size determined by amount of workgroups
		std::vector<typefloat> output_maximum(nr_groups);
		// size of output in bytes
		size_t output_size_max_float = output_maximum.size() * sizeof(typeint);
		// kernel profiler
		cl::Event max_kernel_profiler;
		// reduced max output buffer
		cl::Buffer max_output_buffer(context, CL_MEM_READ_WRITE, output_size_max_float);
		// que kernel
		cl::Kernel kernel_max = cl::Kernel(program, "reduce_max_float");
		// passing input and output buffers
		kernel_max.setArg(0, input_buffer_float);
		kernel_max.setArg(1, max_output_buffer);
		// workers * size of type
		kernel_max.setArg(2, cl::Local(workers_bytes));
		// que kernel
		queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(size_of_temperature_array_padded), cl::NDRange(workers), NULL, &max_kernel_profiler);
		// get kernel output 
		queue.enqueueReadBuffer(max_output_buffer, CL_TRUE, 0, output_size_max_float, &output_maximum[0]);
		// iterating over the reduced maxes to find the max value
		auto max_value = *std::max_element(output_maximum.begin(), output_maximum.end());
		std::cout << "----MAX_FLOAT---- :	" << max_value << endl;

		// ---------------------------------------------------MAX_INT-------------------------------------------------------------

		// output array, size 1 as max are looped over atomically and the min is output in kernel
		std::vector<typeint> output_maximum_int(1);
		// size of output in bytes
		size_t output_size_max_int = output_maximum_int.size() * sizeof(typeint);
		// kernel profiler
		cl::Event max_int_kernel_profiler;
		// output buffer for max value
		cl::Buffer max_int_output_buffer(context, CL_MEM_READ_WRITE, output_size_max_int);
		// que kernel
		cl::Kernel kernel_max_int = cl::Kernel(program, "reduce_max_int");
		// passing input and output buffers
		kernel_max_int.setArg(0, input_buffer_int);
		kernel_max_int.setArg(1, max_int_output_buffer);
		// workers * size of type
		kernel_max_int.setArg(2, cl::Local(workers_int_bytes));
		// que kernel
		queue.enqueueNDRangeKernel(kernel_max_int, cl::NullRange, cl::NDRange(size_of_temperature_array_padded), cl::NDRange(workers), NULL, &max_int_kernel_profiler);
		// get kernel output 
		queue.enqueueReadBuffer(max_int_output_buffer, CL_TRUE, 0, output_size_max_int, &output_maximum_int[0]);
		// getting the max value at [0]
		float max_value_int = output_maximum_int[0];
		std::cout << "----MAX_INT---- :	" << max_value_int << endl << endl;

		// ---------------------------------------------------STANDARDDEVINT-------------------------------------------------------------

		// output array, size 1 as max are looped over atomically and the min is output in kernel
		std::vector<typeint> stdv_output_int(1);
		// size of output in bytes
		size_t output_size_std = stdv_output_int.size() * sizeof(typeint);
		// kernel profiler
		cl::Event stdv_kernel_profiling_event_int;
		// output buffer for sum value	
		cl::Buffer stdv_output_buffer_int(context, CL_MEM_READ_WRITE, output_size_std);
		// que kernel
		cl::Kernel kernel_stdv_int = cl::Kernel(program, "std_dev_int");
		// passing input and output buffers
		kernel_stdv_int.setArg(0, input_buffer_int);
		kernel_stdv_int.setArg(1, stdv_output_buffer_int);
		// passing in mean value for standard deviation calculation
		kernel_stdv_int.setArg(2, sum_int);
		// workers * size of type
		kernel_stdv_int.setArg(3, cl::Local(workers_int_bytes));
		// que kernel
		queue.enqueueNDRangeKernel(kernel_stdv_int, cl::NullRange, cl::NDRange(size_of_temperature_array_padded), cl::NDRange(workers), NULL, &stdv_kernel_profiling_event_int);
		// get kernel output 
		queue.enqueueReadBuffer(stdv_output_buffer_int, CL_TRUE, 0, output_size_std, &stdv_output_int[0]);
		// getting the max value at [0] , dividing to find mean
		float std_dev = float(stdv_output_int[0]) / float(size_of_temperature_array);
		// square rooting value as per standard deviation formula 
		std::cout << "----STANDARD_DEVIATION_INT---- :	" << sqrt(std_dev) << endl;

		// ---------------------------------------------------STANDARDDEVFLOAT-------------------------------------------------------------

		// output array, size determined by amount of workgroups
		std::vector<typefloat> std_dev_float_output(nr_groups);
		// size of output in bytes
		size_t output_size_std_float = std_dev_float_output.size() * sizeof(typeint);
		// kernel profiler
		cl::Event std_dev_float_kernel_profiling_event;
		// reduced sum output buffer	
		cl::Buffer std_dev_float_output_buffer(context, CL_MEM_READ_WRITE, output_size_std_float);
		// que kernel
		cl::Kernel kernel_std_dev_float = cl::Kernel(program, "std_dev_float");
		// passing input and output buffers
		kernel_std_dev_float.setArg(0, input_buffer_float);
		kernel_std_dev_float.setArg(1, std_dev_float_output_buffer);
		// passing in mean value for standard deviation calculation
		kernel_std_dev_float.setArg(2, sum_float);
		// workers * size of type
		kernel_std_dev_float.setArg(3, cl::Local(workers_bytes));
		// que kernel
		queue.enqueueNDRangeKernel(kernel_std_dev_float, cl::NullRange, cl::NDRange(size_of_temperature_array_padded), cl::NDRange(workers), NULL, &std_dev_float_kernel_profiling_event);
		// get kernel output
		queue.enqueueReadBuffer(std_dev_float_output_buffer, CL_TRUE, 0, output_size_std_float, &std_dev_float_output[0]);
		// iterate over output arrays, add reduced sums
		float sum_from_workgroup_float = 0;
		for (std::vector<float>::iterator it = std_dev_float_output.begin(); it != std_dev_float_output.end(); ++it)
			sum_from_workgroup_float += *it;
		float std_dev_float = sum_from_workgroup_float / float(size_of_temperature_array);
		// square rooting value as per standard deviation formula 
		std::cout << "----STANDARD_DEVIATION_FLOAT---- :	" << sqrt(std_dev_float) << endl << endl;

		// ---------------------------------------------------SORTING(IQTRANGE)-------------------------------------------------------------

		// sorted output array, size is same as input
		std::vector<typefloat> output_sort(size_of_temperature_array_padded);
		// size of output in bytes
		size_t output_size_sort = output_sort.size() * sizeof(typefloat);//size in bytes
		// kernel profiler
		cl::Event output_sort_kernel_profiler;
		// sorted array output buffer
		cl::Buffer sort_output_buffer(context, CL_MEM_READ_WRITE, output_size_sort);
		// que kernel
		cl::Kernel kernel_sort = cl::Kernel(program, "selection_sort");
		// passing input and output buffers
		kernel_sort.setArg(0, input_buffer_float);
		kernel_sort.setArg(1, sort_output_buffer);
		// que kernel
		queue.enqueueNDRangeKernel(kernel_sort, cl::NullRange, cl::NDRange(size_of_temperature_array_padded), cl::NDRange(workers), NULL, &output_sort_kernel_profiler);
		// get kernel output
		queue.enqueueReadBuffer(sort_output_buffer, CL_TRUE, 0, output_size_sort, &output_sort[0]);
		std::cout << "----1ST_QUARTILE---- :  " << float(output_sort[size_of_temperature_array_padded/4]) << endl;
		std::cout << "----MEDIAN---- :  " << float(output_sort[size_of_temperature_array_padded / 2]) << endl;
		std::cout << "----3RD_QUARTILE---- :  " << float(output_sort[(size_of_temperature_array_padded / 4) * 3]) << endl << endl;

		// ---------------------------------------------------PROFILING-------------------------------------------------------------

		std::cout << "====EXECUTION_TIMES==== :  " << endl << endl;

		auto copy_buffer_time = copy_buffer_array_profile.getProfilingInfo<CL_PROFILING_COMMAND_END>() - copy_buffer_array_profile.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		
		auto mean_int = sum_kernel_profiling_event_int.getProfilingInfo<CL_PROFILING_COMMAND_END>() - sum_kernel_profiling_event_int.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto mean_float = sum_kernel_profiling_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - sum_kernel_profiling_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto min_int = min_int_kernel_profiler.getProfilingInfo<CL_PROFILING_COMMAND_END>() - min_int_kernel_profiler.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto min_float = min_kernel_profiler.getProfilingInfo<CL_PROFILING_COMMAND_END>() - min_kernel_profiler.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto max_int = max_int_kernel_profiler.getProfilingInfo<CL_PROFILING_COMMAND_END>() - max_int_kernel_profiler.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto max_float = max_kernel_profiler.getProfilingInfo<CL_PROFILING_COMMAND_END>() - max_kernel_profiler.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto standard_dev_float = std_dev_float_kernel_profiling_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - std_dev_float_kernel_profiling_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto standard_dev_int = stdv_kernel_profiling_event_int.getProfilingInfo<CL_PROFILING_COMMAND_END>() - stdv_kernel_profiling_event_int.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto sort_kernel = output_sort_kernel_profiler.getProfilingInfo<CL_PROFILING_COMMAND_END>() - output_sort_kernel_profiler.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto mean_float_atomic = sum_kernel_profiling_event_atomic.getProfilingInfo<CL_PROFILING_COMMAND_END>() - sum_kernel_profiling_event_atomic.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto min_float_atomic = min_kernel_profiler_atomic.getProfilingInfo<CL_PROFILING_COMMAND_END>() - min_kernel_profiler_atomic.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto max_float_atomic = max_kernel_profiler_atomic.getProfilingInfo<CL_PROFILING_COMMAND_END>() - max_kernel_profiler_atomic.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//	auto scan_add_hillis = sum_kernel_profiling_event_hillis.getProfilingInfo<CL_PROFILING_COMMAND_END>() - sum_kernel_profiling_event_hillis.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		
	//	std::cout << "Mean_Hills_Float Temp Kernel Time:	" << scan_add_hillis << " ns" << endl;
		std::cout << "Mean_Int_Reduce Temp Kernel Time:	" << mean_int << " ns" << endl;
		std::cout << "Mean_Float_Reduce_Atomic Temp Kernel Time:	" << mean_float_atomic << " ns" << endl;
		std::cout << "Mean_Float_Reduce Temp Kernel Time:	" << mean_float << " ns" << endl << endl ;
		std::cout << "Min_Int_Reduce Temp Kernel Time:	" << min_int << " ns" << endl;;
		std::cout << "Min_Float_Reduce_Atomic Temp Kernel Time:	" << min_float_atomic << " ns" << endl;;
		std::cout << "Min_Float_Reduce Temp Kernel Time:	" << min_float << " ns" << endl << endl;
		std::cout << "Max_Int_Reduce Temp Kernel Time:	" << max_int << " ns" << endl;;
		std::cout << "Max_Float_Reduce_Atomic Temp Kernel Time:	" << max_float_atomic << " ns" << endl;;
		std::cout << "Max_Float_Reduce Temp Kernel Time:	" << max_float << " ns" << endl << endl;
		std::cout << "Standard Dev_Reduce Float  Kernel Time:	" << standard_dev_float << " ns" << endl;
		std::cout << "Standard Dev_Reduce Int Kernel Time:	" << standard_dev_int << " ns" << endl << endl;
		std::cout << "Sort Kernel Time:	" << sort_kernel << " ns" << endl << endl;
		std::cout << "Total execution time Int reduce	" <<  mean_int + min_int + max_int + standard_dev_int << " ns" << endl;

		std::cout << "Total execution time Float reduce	" << mean_float + min_float + max_float + standard_dev_float<< " ns" << endl << endl;
		std::cout << "Copy to buffer time :	" << copy_buffer_time << " ns" << endl;
		std::cout << "Total execution time program:	" <<  mean_float + min_float + max_float + standard_dev_float + sort_kernel + copy_buffer_time + mean_int + min_int + max_int + standard_dev_int  + mean_float_atomic + min_float_atomic + max_float_atomic << " ns" << endl;

	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}