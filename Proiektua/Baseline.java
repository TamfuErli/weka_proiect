package Proiektua;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.io.FileWriter;
import java.util.Random;

public class Baseline {

    public static void main(String args[]) {
        try {
            //String trainRawPath = args[0];
            String trainBowFss = args[1];
            String hiztegi = args[2];
            String devCsv= args[3];
            String modeloa = args[4];
            String ebaluazioa = args[5];
            String predictionsPath = args[6];
            String devCsvOna = args[7];
            String csvArff = args[8];

//
            DataSource src = new DataSource(trainBowFss);
            Instances train_bow_fss = src.getDataSet();
            train_bow_fss.setClassIndex(0);

            //klase minoritarioa ateratzeko
            int[] classCounts = train_bow_fss.attributeStats(train_bow_fss.classIndex()).nominalCounts;
            if (classCounts == null || classCounts.length == 0) {
                throw new Exception("No class values found");
            }
            int minClassIndex = 0;
            int minCount = Integer.MAX_VALUE;
            for (int n = 0; n < classCounts.length; n++) {
                if (classCounts[n] < minCount) {
                    minCount = classCounts[n];
                    minClassIndex = n;
                }
            }
            Instances devRaw = csvCleanToArff.bihurketaArff(devCsv, devCsvOna, csvArff);
            Instances dev_bow_fss = FssDevTest.applyFssWithDictionary(devRaw, hiztegi);
            dev_bow_fss.setClassIndex(2);
//

            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(train_bow_fss);
            SerializationHelper.write(modeloa, nb);

            try (FileWriter fw = new FileWriter(ebaluazioa)){
                fw.write(" -----------EZ_ZINTZOA---------------\n");
                Evaluation eval = new Evaluation(train_bow_fss);
                fw.write(eval.toMatrixString() + "\n");
                fw.write(("Klasse minoritarioaren f-measure: " + eval.fMeasure(minClassIndex)));

                fw.write(" -----------10_FOLD_CROSS-VALIDATION---------------");
                eval.crossValidateModel(new NaiveBayes(), train_bow_fss, 10, new Random(1));
                fw.write(eval.toMatrixString() + "\n");
                fw.write(("Klasse minoritarioaren f-measure: " + eval.fMeasure(minClassIndex)));

                fw.write(" -----------REPEATED_STRATIFIED_HOLD_OUT---------------\n");
                repeatedStratifiedHoldOut(train_bow_fss, new NaiveBayes(), minClassIndex, ebaluazioa);
                NaiveBayes model = (NaiveBayes) SerializationHelper.read(modeloa);
                predikzioakEgin(predictionsPath, model, dev_bow_fss);
            }
        } catch (Exception e) {
            System.out.println("ERROREA prozesuan");
            e.printStackTrace();
        }
    }

    public static void repeatedStratifiedHoldOut(Instances dataTrain, Classifier model, int minClassIndex, String predictionPath) throws Exception {
        int rep = 5;
        double[][] totalConfusionMatrix = new double[dataTrain.classAttribute().numValues()]
                [dataTrain.classAttribute().numValues()];
        double maxFMeasureTotal = 0.0;
        double ehuneko = 70.0;
        double[] precisions = new double[rep];
        for (int i = 1; i <= rep; i++) {
            Instances datuak = dataTrain;
            datuak.randomize(new Random(i));
            Resample filtroa = new Resample();
            filtroa.setNoReplacement(true);
            filtroa.setSampleSizePercent(ehuneko);
            filtroa.setInputFormat(datuak);
            Instances trainData = Filter.useFilter(datuak, filtroa);

            filtroa.setSampleSizePercent(ehuneko);
            filtroa.setInvertSelection(true);
            filtroa.setInputFormat(datuak);
            Instances devData = Filter.useFilter(datuak, filtroa);
            model.buildClassifier(datuak);

            Evaluation evaluation = new Evaluation(trainData);
            evaluation.evaluateModel(model, devData);
            totalConfusionMatrix = matrizeakGehitu(totalConfusionMatrix, evaluation.confusionMatrix());
            maxFMeasureTotal += evaluation.fMeasure(minClassIndex);
            precisions[i - 1] = evaluation.fMeasure(minClassIndex);
        }

        double[][] avgConfusionMatrix = matrizDividida(totalConfusionMatrix, rep);
        double avgFMeasure = maxFMeasureTotal / rep;
        try (FileWriter fw = new FileWriter(predictionPath, true)){

            fw.write("\n");
            fw.write("Bataz besteko Nahasmen Matrzea");
            for (int i = 0; i < avgConfusionMatrix.length; i++) {
                for (int j = 0; j < avgConfusionMatrix[i].length; j++) {
                    fw.write(avgConfusionMatrix[i][j] + "\t");
                }
                fw.write("\n");
            }
        fw.write("Batez Besteko Max fMeasure : " + avgFMeasure + "\n");
        }
    }

    public static double[][] matrizeakGehitu(double[][] totalMatrix, double[][] nMatrix){
        double[][] resultMatrix = new double[totalMatrix.length][totalMatrix[0].length];
        for(int i=0; i< totalMatrix.length; i++){
            for (int j=0; j<totalMatrix[i].length; j++){
                resultMatrix[i][j] = totalMatrix[i][j] + nMatrix[i][j];
            }
        }
        return resultMatrix;
    }

    public static double[][] matrizDividida(double [][] totalMatrix, int rep){
        double[][] resultMatrix = new double[totalMatrix.length][totalMatrix[0].length];
        for(int i=0; i< totalMatrix.length; i++){
            for (int j=0; j<totalMatrix[i].length; j++){
                resultMatrix[i][j] = totalMatrix[i][j] / rep;
            }
        }
        return resultMatrix;
    }

    public static void predikzioakEgin(String predictionPath, Classifier model,
                                       Instances test_bow_fss) throws Exception {
        try(FileWriter fw = new FileWriter(predictionPath)) {

            for (Instance inst : test_bow_fss) {
                int i = (int) model.classifyInstance(inst);
                inst.setClassValue(test_bow_fss.classAttribute().value(i));
                fw.write(inst + "\n");
            }
        }
    }
}














