package helin_05;

import java.io.File;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Cobweb;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;

/*
 * ����ʽ
 * 
 * ʵ����weka.clusterers.UpdateableClusterer�ӿڵ�Clusterers��������ʽ�ı�ѵ��(��3.5.4�濪ʼ)��
 * �����Խ�ʡ�ڴ棬��Ϊ������Ҫһ���Խ�����ȫ�������ڴ档
 * 
 * ����ѵ��һ������ʽ��clusterer�Ǻܼ򵥵ģ�
 * ����buildClusterer(Instances) ����Instances�����������ݼ��Ľṹ������Instances���������ݣ�Ҳ����û�С�
 * ˳�����updateClusterer(Instances)������ͨ��һ���µ�weka.core.Instances������clusterer��
 * ��ȫ��������������֮�󣬵���updateFinished()����Ϊclusterer��Ҫ���ж���ļ��㡣
 * 
 */
public class WekaUpdateableCluster {
	public static void main(String[] args) throws Exception {
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff"));
		Instances structure = loader.getStructure();// ��ȡ���ݼ��ṹ
		structure.setClassIndex(structure.numAttributes() - 1);

		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-test.arff");
		Instances instancesTest = source.getDataSet(); // ��������ļ�

		Cobweb cluster = new Cobweb();

		Instance current;
		while ((current = loader.getNextInstance(structure)) != null)
			cluster.updateClusterer(current);// ���¾ۺ�����
		cluster.updateFinished();

		ClusterEvaluation eval = new ClusterEvaluation();
		eval.setClusterer(cluster);
		eval.evaluateClusterer(instancesTest);

		System.out.println(eval.getNumClusters());

	}
}
