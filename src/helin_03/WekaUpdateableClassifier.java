package helin_03;

import java.io.File;

import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/*
 * ����ʽ(Incremental)
 * 
 * ʵ����weka.classifiers.UpdateabeClassifier�ӿڵķ�������������ʽ��ѵ���������Խ�Լ�ڴ棬��Ϊ�㲻��Ҫ������һ��ȫ�������ڴ档
 * ����Բ�һ���ĵ�������Щ������ʵ��������ӿڡ�
 * 
 * ����ѧϰһ������ʽ�ķ������Ǻܼ򵥵ģ�
 * ����buildClassifier(Instances)������Instances�����������ݼ��Ľṹ������Instances���������ݣ�Ҳ����û�С�
 * ˳�����updateClassifier(Instances)������ͨ��һ���µ�weka.core.Instances�����·�������
 * 
 */
public class WekaUpdateableClassifier {
	public static void main(String[] args) throws Exception {
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff"));
		Instances structure = loader.getStructure();// ��ȡ���ݼ��ṹ
		structure.setClassIndex(structure.numAttributes() - 1);

		// train NaiveBayes
		NaiveBayesUpdateable nb = new NaiveBayesUpdateable();
		nb.buildClassifier(structure);

		Instance current;
		while ((current = loader.getNextInstance(structure)) != null)
			nb.updateClassifier(current);// ���·�����

		loader = new ArffLoader();
		loader.setFile(new File(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff"));
		Instances instancesTrain = loader.getDataSet();
		instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);

		Evaluation eval = new Evaluation(instancesTrain);
		eval.crossValidateModel(nb, instancesTrain, 10, new Random(1));
		System.out.println(eval.errorRate());
	}
}
