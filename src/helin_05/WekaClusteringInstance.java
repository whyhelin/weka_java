package helin_05;

import java.io.BufferedReader;
import java.io.FileReader;
import weka.clusterers.EM;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaClusteringInstance {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff");
		Instances instancesTrain = source.getDataSet();
		// instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);

		Instances instancesTest = new Instances(
				new BufferedReader(new FileReader(
						"D:\\study\\safety\\weka\\test1\\bank-test.arff")));
		instancesTest.setClassIndex(instancesTest.numAttributes() - 1);

		String[] options = Utils.splitOptions("-I 100");// 迭代100次
		EM cluster = new EM();
		cluster.setOptions(options);
		cluster.buildClusterer(instancesTrain);

		// 设置预测结果
		for (int i = 0; i < instancesTest.numInstances(); i++) {
			int clsRes = cluster.clusterInstance(instancesTest.instance(i));// 返回聚类结果
			System.out.println(clsRes);

			double[] allLabel = cluster.distributionForInstance(instancesTest
					.instance(i));// 返回聚类结果的概率分布

			for (double d : allLabel) {
				System.out.print(d + "---");
			}
			System.out.println();
		}

	}
}
