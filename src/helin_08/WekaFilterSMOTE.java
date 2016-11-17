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
		Instances instancesTrain = source.getDataSet(); // ��������ļ�

		source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-data.arff");
		Instances instancesTest = source.getDataSet(); // ��������ļ�

		instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);
		instancesTest.setClassIndex(instancesTest.numAttributes() - 1);

		Filter remove = new Remove();
		String[] options = Utils.splitOptions("-R 1");
		remove.setOptions(options);// ���ò���
		remove.setInputFormat(instancesTrain);// ����Ӧ�ù�������ʵ������setInputFormat(Instances)�������Ǳ�����Ӧ�ù�����ʱ���һ������

		instancesTrain = Filter.useFilter(instancesTrain, remove);// ����remove���ʵ����
		instancesTest = Filter.useFilter(instancesTest, remove);// ������,Batch
																// filtering

		// ������
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
