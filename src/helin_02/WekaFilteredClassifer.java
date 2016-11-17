package helin_02;

import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

/*
 * 运行中过滤(Filtering on-the-fly)
 * 
 * FilteredClassifier meta-classifier是一种运行中过滤的方式。
 * 它不需要在分类器训练之前先对数据集过滤。并且，在预测的时候，你也不需要将测试数据集再次过滤。
 * 下面的例子中使用meta-classifier with Remove filter和J48，删除一个attribute ID为1的属性。
 */
public class WekaFilteredClassifer {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff");
		Instances instancesTrain = source.getDataSet(); // 读入训练文件

		source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-test.arff");
		Instances instancesTest = source.getDataSet(); // 读入测试文件

		instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);
		instancesTest.setClassIndex(instancesTest.numAttributes() - 1);

		Remove rm = new Remove();
		rm.setAttributeIndices("1"); // 删除第一个属性值，从1开始算，instances集里面还是有这个属性的(参看①处)，只是这个属性不参与训练计算预测。

		J48 j48 = new J48();// classifier
		j48.setUnpruned(true); // using an unpruned J48

		FilteredClassifier fc = new FilteredClassifier();// meta-classifier
		fc.setFilter(rm);
		fc.setClassifier(j48);

		// train and make predictions
		fc.buildClassifier(instancesTrain);

		for (int i = 0; i < instancesTest.numInstances(); i++) {
			double pred = fc.classifyInstance(instancesTest.instance(i));
			System.out.print("ID: " + instancesTest.instance(i).value(0));// ①
			System.out.print(", actual: "
					+ instancesTest.classAttribute().value(
							(int) instancesTest.instance(i).classValue()));
			System.out.println(", predicted: "
					+ instancesTest.classAttribute().value((int) pred));
		}
	}
}
