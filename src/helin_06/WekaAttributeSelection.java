package helin_06;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaAttributeSelection {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff");
		Instances data = source.getDataSet(); // 读入训练文件
		data.setClassIndex(data.numAttributes() - 1);

		AttributeSelection sel = new AttributeSelection();// 直接底层的类

		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);

		sel.setEvaluator(eval);
		sel.setSearch(search);
		sel.SelectAttributes(data);

		int[] indices = sel.selectedAttributes();
		System.out.println(Utils.arrayToString(indices));

		Instances instancesTrain = sel.reduceDimensionality(data);

		// 决策树
		J48 classifier = new J48();
		classifier.buildClassifier(instancesTrain);

		// 10-fold cross-validation
		Evaluation evaluation = new Evaluation(instancesTrain);
		evaluation.crossValidateModel(classifier, instancesTrain, 10,
				new Random(1));
		System.out.println(evaluation.toSummaryString());
	}
}
