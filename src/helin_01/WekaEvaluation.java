package helin_01;

import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class WekaEvaluation {
	public static void main(String[] args) {
		try {
			File inputFile = new File(
					"E:\\study\\machine learning\\projects\\safety\\weka\\test1\\bank-train.arff");// 训练文件
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances instancesTrain = atf.getDataSet(); // 读入训练文件

			inputFile = new File(
					"E:\\study\\machine learning\\projects\\safety\\weka\\test1\\bank-test.arff");// 测试文件
			atf.setFile(inputFile);
			Instances instancesTest = atf.getDataSet(); // 读入测试文件

			instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);// 设置哪一组class为预测的属性
			instancesTest.setClassIndex(instancesTest.numAttributes() - 1);

			// 朴素贝叶斯算法
			Classifier classifier1 = (Classifier) Class.forName(
					"weka.classifiers.bayes.NaiveBayes").newInstance();
			// 决策树
			Classifier classifier2 = (Classifier) Class.forName(
					"weka.classifiers.trees.J48").newInstance();
			// Zero
			Classifier classifier3 = (Classifier) Class.forName(
					"weka.classifiers.rules.ZeroR").newInstance();

			// libsvm
			LibSVM classifier4 = new LibSVM();

			classifier1.buildClassifier(instancesTrain);
			classifier2.buildClassifier(instancesTrain);
			classifier3.buildClassifier(instancesTrain);
			classifier4.buildClassifier(instancesTrain);

			Evaluation eval = new Evaluation(instancesTrain);

			eval.evaluateModel(classifier1, instancesTrain);
			System.out.println(eval.errorRate());
			eval.evaluateModel(classifier2, instancesTest);
			System.out.println(eval.errorRate());
			eval.evaluateModel(classifier3, instancesTest);
			System.out.println(eval.errorRate());
			eval.evaluateModel(classifier4, instancesTest);
			System.out.println(eval.errorRate());

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
