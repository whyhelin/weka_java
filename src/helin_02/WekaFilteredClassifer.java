package helin_02;

import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

/*
 * �����й���(Filtering on-the-fly)
 * 
 * FilteredClassifier meta-classifier��һ�������й��˵ķ�ʽ��
 * ������Ҫ�ڷ�����ѵ��֮ǰ�ȶ����ݼ����ˡ����ң���Ԥ���ʱ����Ҳ����Ҫ���������ݼ��ٴι��ˡ�
 * �����������ʹ��meta-classifier with Remove filter��J48��ɾ��һ��attribute IDΪ1�����ԡ�
 */
public class WekaFilteredClassifer {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff");
		Instances instancesTrain = source.getDataSet(); // ����ѵ���ļ�

		source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-test.arff");
		Instances instancesTest = source.getDataSet(); // ��������ļ�

		instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);
		instancesTest.setClassIndex(instancesTest.numAttributes() - 1);

		Remove rm = new Remove();
		rm.setAttributeIndices("1"); // ɾ����һ������ֵ����1��ʼ�㣬instances�����滹����������Ե�(�ο��ٴ�)��ֻ��������Բ�����ѵ������Ԥ�⡣

		J48 j48 = new J48();// classifier
		j48.setUnpruned(true); // using an unpruned J48

		FilteredClassifier fc = new FilteredClassifier();// meta-classifier
		fc.setFilter(rm);
		fc.setClassifier(j48);

		// train and make predictions
		fc.buildClassifier(instancesTrain);

		for (int i = 0; i < instancesTest.numInstances(); i++) {
			double pred = fc.classifyInstance(instancesTest.instance(i));
			System.out.print("ID: " + instancesTest.instance(i).value(0));// ��
			System.out.print(", actual: "
					+ instancesTest.classAttribute().value(
							(int) instancesTest.instance(i).classValue()));
			System.out.println(", predicted: "
					+ instancesTest.classAttribute().value((int) pred));
		}
	}
}
