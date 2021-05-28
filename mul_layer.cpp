#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/new/mul_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void MulLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	H_ = this->layer_param_.mul_param().output_height();
	CHECK_GT(H_, 0) << "MulLayer output_height must be positive.";
	W_ = bottom[0]->shape(0);
	M_ = bottom[0]->shape(1);
	//CHECK_GT(this->blobs_.size(), 0) << "Blob dose not exit!";
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	}
	else {
		this->blobs_.resize(2);
		// Initialize the weights --
		vector<int> weight_shape;
		weight_shape.push_back(H_);
		weight_shape.push_back(M_);
		this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
		// fill the weights
		shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
			this->layer_param_.mul_param().weight_filler()));
		weight_filler->Fill(this->blobs_[0].get());
		//fill the bias
		vector<int> bias_shape;
		bias_shape.push_back(1);
		bias_shape.push_back(H_);
		this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
		shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
			this->layer_param_.mul_param().bias_filler()));
		bias_filler->Fill(this->blobs_[1].get());
	}
	

	this->param_propagate_down_.resize(this->blobs_.size(), true);
}
template <typename Dtype>
void MulLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	
	vector<int> top_shape;
	top_shape.push_back(W_);
	top_shape.push_back(H_);
	top[0]->Reshape(top_shape);

	vector<int> bias_shape(1, W_);
	bias_multiplier_.Reshape(bias_shape);
	caffe_set(W_, Dtype(1), bias_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void MulLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* weight = this->blobs_[0]->cpu_data();
	const Dtype* bias = this->blobs_[1]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	Blob<Dtype> weight_test;
	vector<int> weight_shape;
	weight_shape.push_back(M_);
	weight_shape.push_back(H_);
	weight_test.Reshape(weight_shape);
	caffe_set(weight_test.count(), Dtype(2), weight_test.mutable_cpu_data());


	caffe_cpu_gemm(CblasNoTrans, CblasTrans, W_, H_, M_, Dtype(1.),
		bottom_data, weight, Dtype(0.), top_data);
	caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, W_, H_, 1, Dtype(1.),
		bias_multiplier_.cpu_data(), bias, Dtype(1.), top_data);
}

template <typename Dtype>
void MulLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	CHECK(!propagate_down[0]) << "Can't backpropagate to MulLayer input.";
	if (this->param_propagate_down_[0]) {
		const Dtype* top_diff = top[0]->cpu_diff();//
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const Dtype* weight = bottom[0]->cpu_data();
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, W_, M_, H_, Dtype(1.),
			top_diff, weight, Dtype(0.), bottom_diff);
	}
}



#ifdef CPU_ONLY
//STUB_GPU(MulLayer);
#endif

INSTANTIATE_CLASS(MulLayer);
REGISTER_LAYER_CLASS(Mul);

}
