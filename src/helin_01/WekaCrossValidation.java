package helin_01;

import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaCrossValidation {
	public static void main(String[] args) throws Exception {
		File inputFile = new File(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff");// 训练文件
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances instancesTrain = atf.getDataSet(); // 读入训练文件

		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-test.arff");
		Instances instancesTest = source.getDataSet(); // 读入测试文件

		instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);
		instancesTest.setClassIndex(instancesTest.numAttributes() - 1);

		// libsvm
		Classifier classifier = new LibSVM();

		classifier.buildClassifier(instancesTrain);
		SerializationHelper.write(
				"D:\\study\\safety\\weka\\test1\\libsvm.model", classifier);
		System.out.println(classifier.classifyInstance(instancesTest
				.instance(5)));// 返回预测的 class 的值。

		Evaluation eval = new Evaluation(instancesTrain);
		eval.crossValidateModel(classifier, instancesTrain, 10, new Random(1));
		System.out.println(eval.errorRate());

		classifier = (Classifier) SerializationHelper
				.read("D:\\study\\safety\\weka\\test1\\libsvm.model");

		classifier.buildClassifier(instancesTrain);
		System.out.println(classifier.classifyInstance(instancesTest
				.instance(5)));// 返回预测的 class 的值。

		eval = new Evaluation(instancesTrain);
		eval.crossValidateModel(classifier, instancesTrain, 10, new Random(1));
		System.out.println(eval.errorRate());
	}
}
