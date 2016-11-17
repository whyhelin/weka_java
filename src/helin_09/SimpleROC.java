package helin_09;

import java.awt.BorderLayout;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

public class SimpleROC {
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(
				"E:\\study\\MachineLearning\\projects\\safety\\weka\\test1\\bank-train.arff");
		Instances instancesTrain = source.getDataSet(); // ����ѵ���ļ�
		instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);

		Classifier cl = new NaiveBayes();
		cl.buildClassifier(instancesTrain);
		Evaluation eval = new Evaluation(instancesTrain);
		eval.crossValidateModel(cl, instancesTrain, 10, new Random(1));
		/*
		 * 3.�������ڵõ�ROC�����AUCֵ��Instances���� ˳����ӡ��һЩ������Ϣ��������SPSS������ROC����
		 * ������ǲ鿴wekaԴ��ͻ�֪�����Instances��������˺ܶ����Ľ����Ϣ
		 * ����:FMeasure��Recall��Precision��True Positive Rate�� False Positive
		 * Rate�ȵȡ����ǿ�������Щ��Ϣ���Ƹ������档
		 */
		ThresholdCurve tc = new ThresholdCurve();
		// classIndex is the index of the class to consider as "positive"
		int classIndex = 0;
		Instances result = tc.getCurve(eval.predictions(), classIndex);
		System.out.println("The area under the ROC��curve: "
				+ eval.areaUnderROC(classIndex));
		/*
		 * ����������ͨ�������ϢInstances����õ�����TP��FP���������� ����������������SPSS��ͨ����ͼ����ROC����
		 */
		int tpIndex = result.attribute(ThresholdCurve.TP_RATE_NAME).index();
		int fpIndex = result.attribute(ThresholdCurve.FP_RATE_NAME).index();
		double[] tpRate = result.attributeToDoubleArray(tpIndex);
		double[] fpRate = result.attributeToDoubleArray(fpIndex);
		/*
		 * 4.ʹ�ý����Ϣinstances��������ʾROC����
		 */
		ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
		// ������AUC�ķ�ʽ������Ĳ�ͬ����ʵ�õ��Ķ���һ����ͬ�Ľ��
		vmc.setROCString("(Area under ROC = "
				+ Utils.doubleToString(ThresholdCurve.getROCArea(result), 4)
				+ ")");
		vmc.setName(result.relationName());
		PlotData2D tempd = new PlotData2D(result);
		tempd.setPlotName(result.relationName());
		tempd.addInstanceNumberAttribute();
		vmc.addPlot(tempd);
		// ��ʾ����
		String plotName = vmc.getName();
		final javax.swing.JFrame jf = new javax.swing.JFrame(
				"Weka Classifier Visualize: " + plotName);
		jf.setSize(500, 400);
		jf.getContentPane().setLayout(new BorderLayout());
		jf.getContentPane().add(vmc, BorderLayout.CENTER);
		jf.addWindowListener(new java.awt.event.WindowAdapter() {
			public void windowClosing(java.awt.event.WindowEvent e) {
				jf.dispose();
			}
		});
		jf.setVisible(true);
	}
}