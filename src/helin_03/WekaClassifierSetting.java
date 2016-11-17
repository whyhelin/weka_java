package helin_03;

import java.io.File;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaClassifierSetting {
	public static void main(String[] args) throws Exception {
		File inputFile = new File(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff");// ѵ���ļ�
		ArffLoader atf = new ArffLoader();
		atf.setFile(inputFile);
		Instances instancesTrain = atf.getDataSet(); // ����ѵ���ļ�

		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-test.arff");
		Instances instancesTest = source.getDataSet(); // ��������ļ�

		instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);
		instancesTest.setClassIndex(instancesTest.numAttributes() - 1);

		// ������
		J48 classifier = new J48();

		String[] options = Utils.splitOptions("-C 0.25 -M 2");
		classifier.setOptions(options);

		classifier.buildClassifier(instancesTrain);

		Evaluation eval = new Evaluation(instancesTrain);
		eval.crossValidateModel(classifier, instancesTrain, 10, new Random(1));
		System.out.println(eval.errorRate());

	}
}
