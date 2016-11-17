package helin_05;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/*
 * ����(Clustering)
 * 
 * ������������ƣ���Ҫ�������������İ����ҵ�weka.clusterers
 * 
 * ����һ��Clusterer
 * ����Batch��
 * 
 */
public class WekaCluster {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff");
		Instances instancesTrain = source.getDataSet(); // ����ѵ���ļ�
		// instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);
		// ���ܴ���class����

		source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-test.arff");
		Instances instancesTest = source.getDataSet(); // ��������ļ�
		// instancesTest.setClassIndex(instancesTest.numAttributes() - 1);

		String[] options = Utils.splitOptions("-I 100");// ����100��
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
