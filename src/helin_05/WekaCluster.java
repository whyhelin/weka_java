package helin_05;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/*
 * 聚类(Clustering)
 * 
 * 聚类与分类相似，必要的类可以在下面的包中找到weka.clusterers
 * 
 * 建立一个Clusterer
 * 批（Batch）
 * 
 */
public class WekaCluster {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff");
		Instances instancesTrain = source.getDataSet(); // 读入训练文件
		// instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);
		// 不能处理class属性

		source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-test.arff");
		Instances instancesTest = source.getDataSet(); // 读入测试文件
		// instancesTest.setClassIndex(instancesTest.numAttributes() - 1);

		String[] options = Utils.splitOptions("-I 100");// 迭代100次
		EM cluster = new EM();
		cluster.setOptions(options);
		cluster.buildClusterer(instancesTrain);

		ClusterEvaluation eval = new ClusterEvaluation();
		eval.setClusterer(cluster);
		eval.evaluateClusterer(instancesTest);

		System.out.println(eval.getNumClusters());
		System.out.println(eval.clusterResultsToString());

	}
}
