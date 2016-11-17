package helin_04;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaStatistics {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(
				"E:\\study\\MachineLearning\\projects\\safety\\weka\\test1\\bank-train.arff");
		Instances instancesTrain = source.getDataSet(); // 读入训练文件
		instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);

		source = new DataSource(
				"E:\\study\\MachineLearning\\projects\\safety\\weka\\test1\\bank-test.arff");
		Instances instancesTest = source.getDataSet(); // 读入测试文件
		instancesTest.setClassIndex(instancesTest.numAttributes() - 1);

		String[] options = Utils.splitOptions("-C 0.25 -M 2");
		J48 classifier = new J48();
		classifier.setOptions(options);
		classifier.buildClassifier(instancesTrain);

		Evaluation eval = new Evaluation(instancesTrain);
		double[] d = eval.evaluateModel(classifier, instancesTest);// 返回预测结果
		for (double pre : d) {
			System.out.println(pre);
		}
		System.out.println(eval.errorRate());
		for (Prediction p : eval.predictions()) {
			System.out.println(p.actual() + "---" + p.predicted() + "---"
					+ p.weight());
		}

		/**
		 * 
		 */
		options = Utils
				.splitOptions("-t E:\\study\\MachineLearning\\projects\\safety\\weka\\test1\\bank-train.arff -T E:\\study\\MachineLearning\\projects\\safety\\weka\\test1\\bank-test.arff");
		System.out.println(Evaluation.evaluateModel(classifier, options));
	}
}
