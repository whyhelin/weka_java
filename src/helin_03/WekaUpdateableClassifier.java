package helin_03;

import java.io.File;

import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/*
 * 增量式(Incremental)
 * 
 * 实现了weka.classifiers.UpdateabeClassifier接口的分类器可以增量式的训练，它可以节约内存，因为你不需要把数据一次全部读入内存。
 * 你可以查一下文档，看哪些分类器实现了这个接口。
 * 
 * 真正学习一个增量式的分类器是很简单的：
 * 调用buildClassifier(Instances)，其中Instances包话这种数据集的结构，其中Instances可以有数据，也可以没有。
 * 顺序调用updateClassifier(Instances)方法，通过一个新的weka.core.Instances，更新分类器。
 * 
 */
public class WekaUpdateableClassifier {
	public static void main(String[] args) throws Exception {
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff"));
		Instances structure = loader.getStructure();// 读取数据集结构
		structure.setClassIndex(structure.numAttributes() - 1);

		// train NaiveBayes
		NaiveBayesUpdateable nb = new NaiveBayesUpdateable();
		nb.buildClassifier(structure);

		Instance current;
		while ((current = loader.getNextInstance(structure)) != null)
			nb.updateClassifier(current);// 更新分类器

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
