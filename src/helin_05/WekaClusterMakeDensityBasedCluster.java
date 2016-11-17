package helin_05;

import java.util.Random;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.MakeDensityBasedClusterer;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/*
 * 在density based clusters这种情况下，
 * 你可用交叉检验的方法去做
 * (注意:用MakeDensityBasedClusterer你可将任何clusterer转换成一下基于密度(density based)的clusterer)。
 * 
 */
public class WekaClusterMakeDensityBasedCluster {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff");
		Instances instancesTrain = source.getDataSet(); // 读入训练文件
		// instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);
		// 不能处理class属性

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
