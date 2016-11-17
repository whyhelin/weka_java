package helin_08;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class WekaFilterSMOTE {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-data.arff");
		Instances instancesTrain = source.getDataSet(); // 读入测试文件

		source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-data.arff");
		Instances instancesTest = source.getDataSet(); // 读入测试文件

		instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);
		instancesTest.setClassIndex(instancesTest.numAttributes() - 1);

		Filter remove = new Remove();
		String[] options = Utils.splitOptions("-R 1");
		remove.setOptions(options);// 设置参数
		remove.setInputFormat(instancesTrain);// 设置应用过滤器的实例集，setInputFormat(Instances)方法总是必需是应用过滤器时最后一个调用

		instancesTrain = Filter.useFilter(instancesTrain, remove);// 返回remove后的实例集
		instancesTest = Filter.useFilter(instancesTest, remove);// 批过滤,Batch
																// filtering

		// 决策树
		J48 classifier = new J48();

		options = Utils.splitOptions("-C 0.25 -M 2");
		classifier.setOptions(options);

		classifier.buildClassifier(instancesTrain);

		Evaluation eval = new Evaluation(instancesTrain);
		eval.evaluateModel(classifier, instancesTest);
		System.out.println(eval.errorRate());

		eval = new Evaluation(instancesTrain);
		eval.crossValidateModel(classifier, instancesTrain, 10, new Random(1));
		System.out.println(eval.errorRate());
	}
}
