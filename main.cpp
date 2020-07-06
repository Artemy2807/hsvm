// ����������
#include <opencv2/opencv.hpp>
#include <exception>
#include "getopt.h"
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <memory>
#include <regex>
#ifdef __linux__
// ��� Linux
#include <unistd.h>
#else
// ��� Windows
#include <windows.h>
#include <algorithm>
#endif

// �������
void program_information(char* program_name);
std::vector<std::string> split(std::string input, std::string regex);
std::vector<std::string> files_in_directory(std::string directory);
std::vector<cv::Mat> load_images(std::string directory, cv::Size image_size);
void compute_hog(std::vector<cv::Mat> image_list, std::vector<cv::Mat>& gradient_list, cv::Size size);
void convert_to_ml(std::vector<cv::Mat> train_samples, cv::Mat& train_data);

// �������� ���
int main(int argc, char **argv) {
	std::string good_image_directory,
		bad_image_directory,
		output_file;
	std::vector<float> labels;
	cv::Size input_size;
	// ����� ��������� ������
	static struct option long_options[] =
	{
		{ "help",   no_argument,       0, 'h' },
		{ "bad", required_argument, 0, 'b' },
		{ "good", required_argument, 0, 'g' },
		{ "size", required_argument, 0, 's' },
		{ "output", required_argument, 0, 'o' },
		{ 0, 0, 0, 0 }
	};
    // ������������ ��������� ��������� ������
	int32_t opt,
		option_index = 0;
	while ((opt = getopt_long(argc, argv, "hb:g:s:o:", long_options, &option_index)) != -1)
	{
		switch (opt)
		{
        // ������� ��������������� ���������
		case 'h':
			program_information(argv[0]);
			return 0;
        // ���� �� ����� � ������������ ����
		case 'b':
			bad_image_directory = optarg;
			break;
        // ���� �� ����� � ������������ ������
		case 'g':
			good_image_directory = optarg;
			break;
        // ������������� ������ ������� ������
		case 's':
		{
			std::vector<std::string> size = split(optarg, ",");
			if (size.size() != 2) break;
			try {
				input_size = cv::Size(std::stoi(size[0]), std::stoi(size[1]));
			}
			catch (std::exception &e) {
				printf("Size must be expressed in numbers. Error in function %s.\n", e.what());
                printf("The -o key is not used correctly. Read the documentation below.\n\n");
			}
		}
		break;
        // �������� ��������� �����
		case 'o':
			output_file = optarg;
			break;
        // ������� ��������������� ���������
		default:
			program_information(argv[0]);
			return 0;
		}
	}
	// ���� ������������ ��������� �� ������������, �� ������� ��������� � ������������� ���������
	if (good_image_directory.empty() || bad_image_directory.empty() || input_size.empty() || output_file.empty()) {
        printf("Use REQUIED ARGUMENT. Read the documentation below.\n\n");
		program_information(argv[0]);
		return 0;
	}
	// ��������� ���������� �� ����������
	std::vector<cv::Mat> good_images = load_images(good_image_directory, input_size),
		bad_images = load_images(bad_image_directory, input_size),
		gradient_list;
    // ��������� ����� ������� ����������
	labels.assign(bad_images.size(), -1);
	labels.insert(labels.end(), good_images.size(), 1);
    // ������������ HOG ���������� ��� ����������
	compute_hog(bad_images, gradient_list, input_size);
	compute_hog(good_images, gradient_list, input_size);
    // �������������� ������ ��� �������� SVM ��������������
	cv::Mat train_data;
	convert_to_ml(gradient_list, train_data);
#if CV_MAJOR_VERSION >= 3
    // ��� OpenCV 3.0.0 � ����
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    // ��������� SVM ��������������
	svm->setType(cv::ml::SVM::EPS_SVR);
	svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
	svm->setP(1.0000000000000001e-01);
	svm->setC(1.0000000000000000e-02);
	svm->setGamma(0.01);
	svm->setNu(0.5);
	svm->setCoef0(0.0);
    // ������� SVM ������������� 
	svm->train(train_data, cv::ml::ROW_SAMPLE, labels);
    // ��������� ��������� SVM �������������
	svm->save(output_file);
#else
    // ��� OpenCV 2.4
	cv::CvSVMParams params;
    // ��������� SVM ��������������
	params.degree = 3;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-3);
	params.kernel_type = cv::CvSVM::LINEAR;
	params.svm_type = cv::CvSVM::EPS_SVR;
	params.C = 1.0000000000000000e-02;
	params.p = 1.0000000000000001e-01;
	params.gamma = 0.01;
	params.nu = 0.5;
	params.coef0 = 0.0;
    // ������� SVM ������������� 
	cv::CvSVM svm;
	svm.train(train_data, cv::Mat(labels), cv::Mat(), cv::Mat(), params);
    // ��������� ��������� SVM �������������
	svm.save(output_file);
#endif
	printf("Program completed successfully\n");
	return 0;
}

void program_information(char* program_name) {
    // ������� ��� ������ ���������������� ���������
	printf(
		"Usage %s [OPTIONS]\n"
		"Options:\n"
		"\t-h program information\n"
		"\t-g (REQUIED ARGUMENT) Path to photos of sign. Example: -g /home/artemy/Good/\n"
		"\t-b (REQUIED ARGUMENT) Path to photos of background. Example: -b /home/artemy/Bad/\n"
		"\t-s (REQUIED ARGUMENT) Data size for SVM. First argument - width, second argument height. Example: -s 64,64\n"
		"\t-o (REQUIED ARGUMENT) Output file name. Example: -o output.xml\n"
		"For example:\n"
		"\t%s -g /home/artemy/Good/ -b /home/artemy/Bad/ -s 64,64 -o output.xml\n",
		program_name, program_name);
}

std::vector<std::string> split(std::string input, std::string regex) {
    // ������� ��� ���������� ������ �� ������ ����� ������ regex
	std::regex re(regex);
	std::sregex_token_iterator
		first{ input.begin(), input.end(), re, -1 },
		last;
	return { first, last };
}

std::vector<std::string> files_in_directory(std::string directory) {
    // ������� ��� ��������� ����������� ���������� (directory) � ������ ����������� � ������ files
	std::vector<std::string> files;
#ifdef __linux__
    // ��� Linux
	char buf[256];
	std::shared_ptr<FILE> pipe(popen(std::string("ls " + directory).c_str(), "r"), pclose);
	while (!feof(pipe.get()))
		if (fgets(buf, 256, pipe.get()) != NULL) {
			std::string file(directory);
			file.append(buf);
			file.pop_back();
			files.push_back(file);
		}
#else
    // ��� Windows
	WIN32_FIND_DATAA FindFileData;
	HANDLE hf;
	std::string windows_directory = directory + "*.*";
	hf = FindFirstFileA(windows_directory.c_str(), &FindFileData);
	if (hf != INVALID_HANDLE_VALUE) {
		do {
			if (std::string(FindFileData.cFileName).length() <= 2) continue;
			files.push_back(directory + std::string(FindFileData.cFileName));
		} while (FindNextFileA(hf, &FindFileData) != 0);
		FindClose(hf);
	}
#endif
	if (files.size() <= 0) exit(0);
	return files;
}

// ���������� �������
std::vector<cv::Mat> load_images(std::string directory, cv::Size image_size) {
    // ������� ��� �������� ����������� �� ����������
	std::vector<cv::Mat> image_list;
	std::vector<std::string> files = files_in_directory(directory);
	for (unsigned int i = 0; i < files.size(); ++i) {
		cv::Mat img = cv::imread(files.at(i));
        // ���� ���� �� ��������, �� ������� ���������
		if (img.empty()) {
			printf("Can not open images with directory: %s\n", files.at(i).c_str());
			continue;
		}
		// �������� ������ � ��������� � ������
		cv::resize(img, img, image_size);
		image_list.push_back(img.clone());
	}
	return image_list;
}

void compute_hog(std::vector<cv::Mat> image_list, std::vector<cv::Mat>& gradient_list, cv::Size size) {
    // ������� ��� ���������� HOG �����������
	cv::HOGDescriptor hog;
	hog.winSize = size;
	cv::Mat gray;
	std::vector<float> descriptors;
	for (unsigned int i = 0; i < image_list.size(); i++) {
		cvtColor(image_list[i], gray, cv::COLOR_BGR2GRAY);
		hog.compute(gray, descriptors, cv::Size(8, 8), cv::Size(0, 0));
		gradient_list.push_back(cv::Mat(descriptors).clone());
	}
}

void convert_to_ml(std::vector<cv::Mat> train_samples, cv::Mat& train_data) {
    // ������� ��������������� ������ ��� �������� SVM ��������������
	int rows = (int)train_samples.size(),
		cols = 0;
#ifdef __linux__
	cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
#else
	cols = (int)max(train_samples[0].cols, train_samples[0].rows);
#endif
	cv::Mat tmp(1, cols, CV_32FC1);
	train_data = cv::Mat(rows, cols, CV_32FC1);
	std::vector<cv::Mat>::const_iterator itr = train_samples.begin();
	std::vector<cv::Mat>::const_iterator end = train_samples.end();
	for (int i = 0; itr != end; ++itr, ++i) {
		CV_Assert(itr->cols == 1 ||
			itr->rows == 1);
		if (itr->cols == 1) {
			transpose(*(itr), tmp);
			tmp.copyTo(train_data.row(i));
		}
		else if (itr->rows == 1) itr->copyTo(train_data.row(i));
	}
}
