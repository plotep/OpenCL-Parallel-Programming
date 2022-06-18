# OpenCL-Parallel-Programming
The program implements the basic functionalities of calculating mean, min maxand standard deviation for ints and floats using OpenCL.
The hillis-steele implementation for mean calculation and the structure of the program has been adapted from Tutorial 3 - https://github.com/alanmillard/OpenCL-Tutorials
Atomic functions have been made for the float values, adapted from : https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/ .
Sorting was implement using the selection sort algorithm, found on : http://www.bealto.com/gpu-sorting_parallel-selection.html . This has been changed so that it 
fits the structure of the temperature statistics program. From the sorted output array, the quartile values are calculated and shown. Kernel runtimes are reported
for both variable types, the memory buffer transfer time is recorded and shown in the console output, and the total int and float operation runtimes are recorded.
The total program runtime is also recorded and shown.
## Results
#[result image](https://github.com/plotep/OpenCL-Parallel-Programming/blob/main/Images/Results.png?raw=true)
