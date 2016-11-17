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
 * ����ѡ��(Attribute selection)
 * 
 * meta-classifier
 * 
 * ��ʵû�б�Ҫ����Ĵ�����ֱ��ʹ������ѡ���࣬��Ϊ�Ѿ���meta-classifier��filter���Խ�������ѡ��
 * ����Ϊ�������ԣ��ײ�ķ�����Ȼ���г����ˡ�
 * ���������CfsSubsetEVal��GreedStepwise���������ӡ�
 * 
 */
public class WekaAttributeSelectedClassifier {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(
				"D:\\study\\safety\\weka\\test1\\bank-train.arff");
		Instances data = source.getDataSet(); // ����ѵ���ļ�
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
