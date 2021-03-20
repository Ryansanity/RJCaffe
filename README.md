# RJCaffe
add mul_layer edit by owner.
该层主要功能是实现矩阵乘法，bottom的一个参数是输入矩阵的宽W_，第二个参数是输入矩阵的高M_，top的参数是输出矩阵的高H_，支持bias。计算过程直接调用caffe_cpu_gemm即可。
