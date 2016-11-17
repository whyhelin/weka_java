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
					"E:\\study\\machine learning\\projects\\safety\\weka\\test1\\bank-train.arff");// ѵ���ļ�
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances instancesTrain = atf.getDataSet(); // ����ѵ���ļ�

			inputFile = new File(
					"E:\\study\\machine learning\\projects\\safety\\weka\\test1\\bank-test.arff");// �����ļ�
			atf.setFile(inputFile);
			Instances instancesTest = atf.getDataSet(); // ��������ļ�

			instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);// ������һ��classΪԤ�������
			instancesTest.setClassIndex(instancesTest.numAttributes() - 1);

			// ���ر�Ҷ˹�㷨
			Classifier classifier1 = (Classifier) Class.forName(
					"weka.classifiers.bayes.NaiveBayes").newInstance();
			// ������
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
