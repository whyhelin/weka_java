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
		Instances instancesTrain = source.getDataSet(); // 读入训练文件
		instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);

		Classifier cl = new NaiveBayes();
		cl.buildClassifier(instancesTrain);
		Evaluation eval = new Evaluation(instancesTrain);
		eval.crossValidateModel(cl, instancesTrain, 10, new Random(1));
		/*
		 * 3.生成用于得到ROC曲面和AUC值的Instances对象 顺带打印了一些其它信息，用于在SPSS中生成ROC曲面
		 * 如果我们查看weka源码就会知道这个Instances对象包含了很多分类的结果信息
		 * 例如:FMeasure、Recall、Precision、True Positive Rate、 False Positive
		 * Rate等等。我们可以用这些信息绘制各种曲面。
		 */
		ThresholdCurve tc = new ThresholdCurve();
		// classIndex is the index of the class to consider as "positive"
		int classIndex = 0;
		Instances result = tc.getCurve(eval.predictions(), classIndex);
		System.out.println("The area under the ROC　curve: "
				+ eval.areaUnderROC(classIndex));
		/*
		 * 在这里我们通过结果信息Instances对象得到包含TP、FP的两个数组 这两个数组用于在SPSS中通过线图绘制ROC曲面
		 */
		int tpIndex = result.attribute(ThresholdCurve.TP_RATE_NAME).index();
		int fpIndex = result.attribute(ThresholdCurve.FP_RATE_NAME).index();
		double[] tpRate = result.attributeToDoubleArray(tpIndex);
		double[] fpRate = result.attributeToDoubleArray(fpIndex);
		/*
		 * 4.使用结果信息instances对象来显示ROC曲面
		 */
		ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
		// 这个获得AUC的方式与上面的不同，其实得到的都是一个共同的结果
		vmc.setROCString("(Area under ROC = "
				+ Utils.doubleToString(ThresholdCurve.getROCArea(result), 4)
				+ ")");
		vmc.setName(result.relationName());
		PlotData2D tempd = new PlotData2D(result);
		tempd.setPlotName(result.relationName());
		tempd.addInstanceNumberAttribute();
		vmc.addPlot(tempd);
		// 显示曲面
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