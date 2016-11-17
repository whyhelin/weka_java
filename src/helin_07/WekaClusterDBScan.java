package helin_07;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.DBSCAN;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/*
 * DBSCAN算法是一个基于密度的聚类方法，
 * 优点：
 * 1、  形成的簇可以有任意的形状；
 * 2、  自动确定簇的数目；
 * 3、  可以分离簇和噪声数据；
 * 4、  可以被空间索引结构支持；
 * 5、  效率高，对大量数据也是如此；
 * 6、  一次扫面数据即可完成聚类；
 * 
 * 原理：DBSCAN是基于密度的，
 * 这里基于密度是表示在以某个点为长度的邻域内，所包含的点的个数；
 * 于是所分成的簇中的每一个点都会在簇中的某一个点的邻域内；
 * 如果某一个点不在任何簇的邻域内，那么这样的点就被成为噪声点。
 * DBSCAN的密度用点的ε邻域中所包含的点数MinPts来表示；
 * 距离可根据需要选择一种距离的表示方法。
 * 
 */
public class WekaClusterDBScan {
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

		DBSCAN cluster = new DBSCAN();
		cluster.setEpsilon(3);// 设置半径
		cluster.setMinPoints(4);// 设置最小点数
		cluster.setDistanceFunction(new EuclideanDistance());// 设置距离方式
		cluster.buildClusterer(instancesTrain);

		ClusterEvaluation eval = new ClusterEvaluation();
		eval.setClusterer(cluster);
		eval.evaluateClusterer(instancesTest);

		System.out.println(eval.getNumClusters());
		System.out.println(eval.clusterResultsToString());

	}
}
