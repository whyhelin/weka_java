package helin_06;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Debug.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

/*
 * Filter
 * 
 * 过滤器方法是很直接的，在设置过滤器之后，你就可以通过过滤器过滤并得到过滤后的数据集。
 */
public class WekaAttributeSelection_filter {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff");
		Instances data = source.getDataSet(); // 读入训练文件
		data.setClassIndex(data.numAttributes() - 1);

		AttributeSelection filter = new AttributeSelection();// filter

		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);

		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(data);

		Instances instancesTrain = Filter.useFilter(data, filter);
		System.out.println(instancesTrain);

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
