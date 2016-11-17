package helin_05;

import java.util.Random;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.MakeDensityBasedClusterer;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/*
 * ��density based clusters��������£�
 * ����ý������ķ���ȥ��
 * (ע��:��MakeDensityBasedClusterer��ɽ��κ�clustererת����һ�»����ܶ�(density based)��clusterer)��
 * 
 */
public class WekaClusterMakeDensityBasedCluster {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff");
		Instances instancesTrain = source.getDataSet(); // ����ѵ���ļ�
		// instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);
		// ���ܴ���class����

		MakeDensityBasedClusterer cluster = new MakeDensityBasedClusterer();
		cluster.buildClusterer(instancesTrain);

		System.out.println(ClusterEvaluation.crossValidateModel(cluster,
				instancesTrain, 10, new Random(1)));

		String[] options = Utils
				.splitOptions("-t D:\\study\\safety\\weka\\test1\\bank-train.arff -T D:\\study\\safety\\weka\\test1\\bank-test.arff");
		System.out.println(ClusterEvaluation
				.evaluateClusterer(cluster, options));
	}
}
