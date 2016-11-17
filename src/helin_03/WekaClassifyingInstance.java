package helin_03;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/*
 * 分类样本(classifying instances)
 * 
 * 如果你想用你新训练的分类器去分类一个未标记数据集(unlabeled dataset)，
 * 你可以使用下面的代码段，它从../unlabeled.arff中读取数据，
 * 并用先前训练的分类器tree去标记样本，并保存标记样本在../labeled.arff中
 * 
 */
public class WekaClassifyingInstance {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff");
		Instances instancesTrain = source.getDataSet();
		instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);

		Instances unlabeled = new Instances(new BufferedReader(new FileReader(
				"D:\\study\\safety\\weka\\test1\\bank-test-unlabeled.arff")));
		unlabeled.setClassIndex(unlabeled.numAttributes() - 1);

		Instances labeled = new Instances(unlabeled);// 复制一份拷贝

		// 决策树
		J48 classifier = new J48();

		String[] options = Utils.splitOptions("-C 0.25 -M 2");
		classifier.setOptions(options);

		classifier.buildClassifier(instancesTrain);

		// 设置预测结果
		for (int i = 0; i < unlabeled.numInstances(); i++) {
			double clsLabel = classifier
					.classifyInstance(unlabeled.instance(i));// 返回分类结果
			labeled.instance(i).setClassValue(clsLabel);
			System.out
					.println(unlabeled.classAttribute().value((int) clsLabel));

			double[] allLabel = classifier.distributionForInstance(unlabeled
					.instance(i));// 返回分类结果的概率分布

			for (double d : allLabel) {
				System.out.print(d);
			}
			System.out.println();
		}

		// 保存标记结果
		BufferedWriter writer = new BufferedWriter(new FileWriter(
				"D:\\study\\safety\\weka\\test1\\bank-result-labeled.arff"));
		writer.write(labeled.toString());
		writer.newLine();
		writer.flush();
		writer.close();
	}
}
