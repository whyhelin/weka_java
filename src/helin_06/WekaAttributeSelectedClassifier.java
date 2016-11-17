package helin_06;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Debug.Random;
import weka.core.converters.ConverterUtils.DataSource;

/*
 * 属性选择(Attribute selection)
 * 
 * meta-classifier
 * 
 * 其实没有必要在你的代码中直接使用属性选择类，因为已经有meta-classifier和filter可以进行属性选择，
 * 但是为了完整性，底层的方法仍然被列出来了。
 * 下面就是用CfsSubsetEVal和GreedStepwise方法的例子。
 * 
 */
public class WekaAttributeSelectedClassifier {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff");
		Instances data = source.getDataSet(); // 读入训练文件
		data.setClassIndex(data.numAttributes() - 1);

		AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();

		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);

		J48 base = new J48();
		classifier.setClassifier(base);
		classifier.setEvaluator(eval);
		classifier.setSearch(search);

		// 10-fold cross-validation
		Evaluation evaluation = new Evaluation(data);
		evaluation.crossValidateModel(classifier, data, 10, new Random(1));
		System.out.println(evaluation.toSummaryString());
	}
}
