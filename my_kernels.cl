void AtomicAdd_g_f(volatile __global float* input, const float output) {

	// reference https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
	union { unsigned int intVal;
			float floatVal;
	} newVal;
	union {
		unsigned int intVal;
		float floatVal;
	} prevVal;
	do {
		prevVal.floatVal = *input;
		newVal.floatVal = prevVal.floatVal + output;
	} while (atomic_cmpxchg((volatile __global unsigned int*)input, prevVal.intVal,newVal.intVal) != prevVal.intVal);
}
void AtomicMin_g_f(volatile __global float* input, const float output) {

	// reference https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
	union {
		unsigned int intVal;
		float floatVal;
	} newVal;
	union {
		unsigned int intVal;
		float floatVal;
	} prevVal;
	do {
		prevVal.floatVal = *input;
		newVal.floatVal = min(prevVal.floatVal, output);
	} while (atomic_cmpxchg((volatile __global unsigned int*)input, prevVal.intVal,
		newVal.intVal) != prevVal.intVal);
}
void AtomicMax_g_f(volatile __global float* input, const float output) {

	// reference https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
	union {
		unsigned int intVal;
		float floatVal;
	} newVal;
	union {
		unsigned int intVal;
		float floatVal;
	} prevVal;
	do {
		prevVal.floatVal = *input;
		newVal.floatVal = max(prevVal.floatVal, output);
	} while (atomic_cmpxchg((volatile __global unsigned int*)input, prevVal.intVal,
		newVal.intVal) != prevVal.intVal);
}





kernel void reduce_add_int(global const int* input, global int* output, local int* scratch)
{
	// global globalID of a work item
	int globalID = get_global_id(0);
	// local id of a work item
	int localID = get_local_id(0);
	// amount of workers
	int workersize = get_local_size(0);	
	// copy input values from global memory to local memory
	scratch[localID] = input[globalID];
	// sync items
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int stride = workersize / 2; stride > 0; stride /= 2) {
		// sync items
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if (localID < stride) {
			// add partial sum
			scratch[localID] += scratch[localID + stride];
		}
	}
	// if at the bottom of the reduction tree of workgroup, add the current workgroups sum to total
	if (!localID) {
		atomic_add(&output[0], scratch[localID]);
	}
}

kernel void reduce_add_float(global const float* input, global float* output, local float* scratch) {
	// global globalID of a work item
	int globalID = get_global_id(0);
	// local id of a work item
	int localID = get_local_id(0);
	// amount of workers
	int workersize = get_local_size(0);
	// workgroup ID = number of total items / workers
	const int workgroupID = globalID / workersize;
	// copy input values from global memory to local memory
	scratch[localID] = input[globalID];
	
	for (int stride = workersize / 2; stride > 0; stride /= 2) {
		// sync items
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if (localID < stride) {
			// add partial sum
			scratch[localID] += scratch[localID + stride];
		}
	}
	// if at the bottom of the reduction tree of workgroup, write results to output array
	if (!localID) {	
		output[workgroupID] = scratch[0];
//		AtomicAdd_g_f(&output[0], scratch[localID]);
	}
}
kernel void reduce_add_float_atomic(global const float* input, global float* output, local float* scratch) {
	// global globalID of a work item
	int globalID = get_global_id(0);
	// local id of a work item
	int localID = get_local_id(0);
	// amount of workers
	int workersize = get_local_size(0);
	// workgroup ID = number of total items / workers
	const int workgroupID = globalID / workersize;
	// copy input values from global memory to local memory
	scratch[localID] = input[globalID];

	for (int stride = workersize / 2; stride > 0; stride /= 2) {
		// sync items
		barrier(CLK_LOCAL_MEM_FENCE);

		if (localID < stride) {
			// add partial sum
			scratch[localID] += scratch[localID + stride];
		}
	}
	// if at the bottom of the reduction tree of workgroup, write results to output array
	if (!localID) {
		AtomicAdd_g_f(&output[0], scratch[localID]);
	}
}


kernel void reduce_min_int(global const int* input, global int* output, local int* scratch)
{
	// global globalID of a work item
	int globalID = get_global_id(0);
	// local id of a work item
	int localID = get_local_id(0);
	// amount of workers
	int workersize = get_local_size(0);
	// copy input values from global memory to local memory
	scratch[localID] = input[globalID];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int stride = workersize / 2; stride > 0; stride /= 2) {
		// sync items
		barrier(CLK_LOCAL_MEM_FENCE);	

		if (localID < stride) {

			if (scratch[localID + stride] < scratch[localID])
			{
				// only keep the min value from each reduction step
				scratch[localID] = scratch[localID + stride];
			}

		}
	}
	// if at the bottom of the reduction tree of workgroup, compare the previous workgroups min with next workgroup, only keep the lowest number
	if (!localID) {
		atomic_min(&output[0], scratch[localID]);
	}
}
kernel void reduce_min(global const float* input, global float* output, local float* scratch)
{
	// global globalID of a work item
	int globalID = get_global_id(0);
	// local id of a work item
	int localID = get_local_id(0);
	// amount of workers
	int workersize = get_local_size(0);
	// workgroup ID = number of total items / workers
	const int workgroupID = globalID / workersize;

	scratch[localID] = input[globalID];

	barrier(CLK_LOCAL_MEM_FENCE);


	for (int stride = workersize / 2; stride > 0; stride /= 2) {

		barrier(CLK_LOCAL_MEM_FENCE);
		
		if (localID < stride) {

			if (scratch[localID + stride] < scratch[localID])
			{
				// only keep the min value from each reduction step
				scratch[localID] = scratch[localID + stride];
			}

		}
	}
	// if at the bottom of the reduction tree of workgroup, save minimum found in each reduction to array
	if (!localID) {
		output[workgroupID] = scratch[0];
	}
}
kernel void reduce_min_atomic(global const float* input, global float* output, local float* scratch)
{
	// global globalID of a work item
	int globalID = get_global_id(0);
	// local id of a work item
	int localID = get_local_id(0);
	// amount of workers
	int workersize = get_local_size(0);
	// workgroup ID = number of total items / workers
	const int workgroupID = globalID / workersize;

	scratch[localID] = input[globalID];

	barrier(CLK_LOCAL_MEM_FENCE);


	for (int stride = workersize / 2; stride > 0; stride /= 2) {

		barrier(CLK_LOCAL_MEM_FENCE);

		if (localID < stride) {

			if (scratch[localID + stride] < scratch[localID])
			{
				// only keep the min value from each reduction step
				scratch[localID] = scratch[localID + stride];
			}

		}
	}
	// if at the bottom of the reduction tree of workgroup, save minimum found in each reduction to array
	if (!localID) {
		AtomicMin_g_f(&output[0], scratch[localID]);
	}
}

// Reduce Max value given in vector input outputted in vector output via local memory vector Scratch
kernel void reduce_max_int(global const int* input, global int* output, local int* scratch)
{
	// global globalID of a work item
	int globalID = get_global_id(0);
	// local id of a work item
	int localID = get_local_id(0);
	// amount of workers
	int workersize = get_local_size(0);
	// copy input values from global memory to local memory
	scratch[localID] = input[globalID];


	barrier(CLK_LOCAL_MEM_FENCE);

	for (int stride = workersize / 2; stride > 0; stride /= 2) {

		barrier(CLK_LOCAL_MEM_FENCE);
		
		if (localID < stride) {

			if (scratch[localID + stride] > scratch[localID])
			{
				// only keep the max value from each reduction step
				scratch[localID] = scratch[localID + stride];
			}

		}
	}
	// if at the bottom of the reduction tree of workgroup, compare the previous workgroups max with next workgroup, only keep the highest number
	if (!localID) {
		atomic_max(&output[0], scratch[localID]);
	}
}

kernel void reduce_max_atomic(global const float* input, global float* output, local float* scratch)
{
	// global globalID of a work item
	int globalID = get_global_id(0);
	// local id of a work item
	int localID = get_local_id(0);
	// amount of workers
	int workersize = get_local_size(0);
	// workgroup ID = number of total items / workers
	const int workgroupID = globalID / workersize;
	// copy input values from global memory to local memory
	scratch[localID] = input[globalID];


	barrier(CLK_LOCAL_MEM_FENCE);


	for (int stride = workersize / 2; stride > 0; stride /= 2) {

		barrier(CLK_LOCAL_MEM_FENCE);
		
		if (localID < stride) {

			if (scratch[localID + stride] > scratch[localID])
			{
				// only keep the max value from each reduction step
				scratch[localID] = scratch[localID + stride];
			}

		}
	}

	// if at the bottom of the reduction tree of workgroup, save max found in each reduction to array
	if (!localID) {
		AtomicMax_g_f(&output[0], scratch[localID]);
	}
}
kernel void reduce_max_float(global const float* input, global float* output, local float* scratch)
{
	// global globalID of a work item
	int globalID = get_global_id(0);
	// local id of a work item
	int localID = get_local_id(0);
	// amount of workers
	int workersize = get_local_size(0);
	// workgroup ID = number of total items / workers
	const int workgroupID = globalID / workersize;
	// copy input values from global memory to local memory
	scratch[localID] = input[globalID];


	barrier(CLK_LOCAL_MEM_FENCE);


	for (int stride = workersize / 2; stride > 0; stride /= 2) {

		barrier(CLK_LOCAL_MEM_FENCE);

		if (localID < stride) {

			if (scratch[localID + stride] > scratch[localID])
			{
				// only keep the max value from each reduction step
				scratch[localID] = scratch[localID + stride];
			}

		}
	}

	// if at the bottom of the reduction tree of workgroup, save max found in each reduction to array
	if (!localID) {
		output[workgroupID] = scratch[0];
	}
}
// Returns Vector containing the Standard Deviation of the Sum input
kernel void std_dev_int(global const int* input, global int* output, float mean, local int* scratch)
{
	// global globalID of a work item
	int globalID = get_global_id(0);
	// local id of a work item
	int localID = get_local_id(0);
	// amount of workers
	int workersize = get_local_size(0);
	
	// calculate square
	scratch[localID] = ((input[globalID] - mean) * (input[globalID] - mean));

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int stride = workersize / 2; stride > 0; stride /= 2) {

		barrier(CLK_LOCAL_MEM_FENCE);
		
		if (localID < stride) {
			// add partial sum
			scratch[localID] += scratch[localID + stride];
		}
	}

	// if at the bottom of the reduction tree of workgroup, add the current workgroups sum to total
	if (!localID) {
		atomic_add(&output[0], scratch[localID]);
	}
		

}
kernel void std_dev_float(global const float* input, global float* output, float mean, local float* scratch)
{
	// global globalID of a work item
	int globalID = get_global_id(0);
	// local id of a work item
	int localID = get_local_id(0);
	// amount of workers
	int workersize = get_local_size(0);
	// workgroup ID = number of total items / workers
	const int workgroupID = globalID / workersize;
	// calculate square
	scratch[localID] = ((input[globalID] - mean) * (input[globalID] - mean));


	barrier(CLK_LOCAL_MEM_FENCE);


	for (int stride = workersize / 2; stride > 0; stride /= 2) {

		barrier(CLK_LOCAL_MEM_FENCE);	
		if (localID < stride) {
			// add partial sum
			scratch[localID] += scratch[localID + stride];
		}
	}

	// if at the bottom of the reduction tree of workgroup, write results to output array
	if (!localID) {
		output[workgroupID] = scratch[0];
	}
		

}


// selection sort , reference : http://www.bealto.com/gpu-sorting_parallel-selection.html
kernel void selection_sort(__global const float* input, __global float* output)
{
	int globalID = get_global_id(0);
	int workersize = get_global_size(0);

	float ikey = input[globalID];

	int pos = 0;
	for (int j = 0; j < workersize; j++)
	{
		float jkey = input[j];
		bool smaller = (jkey < ikey) || (jkey == ikey && j < globalID);
		pos += (smaller) ? 1 : 0;
	}
	output[pos] = ikey;
}

kernel void Hillis_Steele(global const float* input, global float* output, local float* scratch_1, local float* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// used to as a temporary variable for bufer swaps
	local float* scratch_3;

	// store from global to local memory
	scratch_1[lid] = input[id];

	// wait for threads 
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
		{
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		}
		else
		{
			scratch_2[lid] = scratch_1[lid];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// swap sorted buffers
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	// copy to output array
	output[id] = scratch_1[lid];
}
kernel void scan_bl(global int* A , global float* output) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride * 2)) == 0)
			A[id] += A[id - stride];
		output[id] = A[id];
		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N - 1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N / 2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride * 2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;		 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
	

}
__kernel void blelloch_scan(__global const float* input,__global float* output,__global float* group_sum,__local float* temp)
{
	uint global_id = get_global_id(0);
	uint local_id = get_local_id(0);

	uint group_id = get_group_id(0);
	uint group_size = get_local_size(0);

	uint depth = 1;

	if (global_id >= get_global_size(0)) return; // if at the end of the array , stop

	temp[local_id] = input[global_id]; // loading in data from global to local memory

	//upsweep
	for (uint stride = group_size >> 1; stride > 0; stride >>= 1) {
		// syncing
		barrier(CLK_LOCAL_MEM_FENCE);

		if (local_id < stride) {
			uint i = depth * (2 * local_id + 1) - 1;
			uint j = depth * (2 * local_id + 2) - 1;
			temp[j] += temp[i];
		}

		depth <<= 1;
	}

	if (local_id == 0) {
		group_sum[group_id] = temp[group_size - 1];
		temp[group_size - 1] = 0; // exclusive scan
	}
	//downsweep
	for (uint stride = 1; stride < group_size; stride <<= 1) {

		depth >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);

		if (local_id < stride) {
			uint i = depth * (2 * local_id + 1) - 1;
			uint j = depth * (2 * local_id + 2) - 1;

			float t = temp[j];
			temp[j] += temp[i];
			temp[i] = t;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	output[global_id] = temp[local_id];
}