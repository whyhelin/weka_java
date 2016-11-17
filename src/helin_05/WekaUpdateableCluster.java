package helin_05;

import java.io.File;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Cobweb;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;

/*
 * 增量式
 * 
 * 实现了weka.clusterers.UpdateableClusterer接口的Clusterers可以增量式的被训练(从3.5.4版开始)。
 * 它可以节省内存，因为它不需要一次性将数据全部读入内存。
 * 
 * 真正训练一个增量式的clusterer是很简单的：
 * 调用buildClusterer(Instances) 其中Instances包话这种数据集的结构，其中Instances可以有数据，也可以没有。
 * 顺序调用updateClusterer(Instances)方法，通过一个新的weka.core.Instances，更新clusterer。
 * 当全部样本被处理完之后，调用updateFinished()，因为clusterer还要进行额外的计算。
 * 
 */
public class WekaUpdateableCluster {
	public static void main(String[] args) throws Exception {
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff"));
		Instances structure = loader.getStructure();// 读取数据集结构
		structure.setClassIndex(structure.numAttributes() - 1);

		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-test.arff");
		Instances instancesTest = source.getDataSet(); // 读入测试文件

		Cobweb cluster = new Cobweb();

		Instance current;
		while ((current = loader.getNextInstance(structure)) != null)
			cluster.updateClusterer(current);// 更新聚合类器
		cluster.updateFinished();

		ClusterEvaluation eval = new ClusterEvaluation();
		eval.setClusterer(cluster);
		eval.evaluateClusterer(instancesTest);

		System.out.println(eval.getNumClusters());

	}
}
